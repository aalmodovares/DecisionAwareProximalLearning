import torch
from torch import distributions as D
import lightning as L
from typing import List, Tuple, Sequence, Union

from zuko.flows import Flow
from causalflows.flows import CausalFlow


class DeCaFlow(L.LightningModule):
    def __init__(self,
                 encoder: Union[Flow, None],
                 flow: CausalFlow,
                 regularize: bool,
                 warmup: int = None,
                 lr: float = 1e-3,
                 optimizer_cls: type = torch.optim.Adam,
                 optimizer_kwargs: dict = None,
                 scheduler_cls: type = None,
                 scheduler_kwargs: dict = None,
                 scheduler_monitor: str = None,
                 conditional_vars: int = 0,
                 prior_conditioner: Union[Flow, None] = None):
        '''
        :param encoder: Conditional Normalizing Flow encoder. Can be causal or non-causal.
          If None, the model is a Causal Normalizing Flow
        :param flow: Causal Normalizing flow decoder
        :param regularize: bool, apply KL regularization to avoid posterior collapse
        :param warmup: int, number of epochs to warm up the KL regularization
        :param conditional_vars: int, number of conditional variables
        '''
        super().__init__()
        self.encoder = encoder
        self.flow = flow # decoder
        self.automatic_optimization = False

        self.regularize = regularize
        self.warmup = warmup

        if self.regularize:
            assert self.warmup is not None, "Regularization is set to True, but warmup is not set."

        # Optimizer and scheduler configuration
        self.lr = lr
        self.optimizer_cls = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs or {}
        self.scheduler_cls = scheduler_cls
        self.scheduler_kwargs = scheduler_kwargs or {}
        self.scheduler_monitor = scheduler_monitor

        # modification for conditional vars
        self.conditional_vars = conditional_vars

        self.prior_conditioner = prior_conditioner

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self.parameters(),
            lr=self.lr,
            **self.optimizer_kwargs
        )

        if self.scheduler_cls is not None:
            scheduler = self.scheduler_cls(optimizer, **self.scheduler_kwargs)
            scheduler_config = {
                "scheduler": scheduler,
                "monitor": self.scheduler_monitor,
                "interval": "epoch",
                "frequency": 1,
                "strict": True
            }
            return [optimizer], [scheduler_config]

        return optimizer

    def sample_prior(self, sample_shape: torch.Size,
                     conditional_samples: torch.Tensor | None = None # modification for conditional samples
                     ) -> torch.Tensor:
        if self.conditional_vars>0:
            assert conditional_samples is not None
            if self.prior_conditioner is None:
                p_z = D.Independent(
                    D.Normal(loc=torch.zeros(self.encoder.latent_dim).to(self.device), scale=1),
                    reinterpreted_batch_ndims=1)
                sample_shape = (len(conditional_samples),)
                return p_z.sample(sample_shape)
            else:
                p_z = self.prior_conditioner(conditional_samples)
                return p_z.sample() # if conditional samples included, no sample shape is taken into account
        else:
            p_z= D.Independent(
                D.Normal(loc=torch.zeros(self.encoder.latent_dim).to(self.device), scale=1),
                reinterpreted_batch_ndims=1)
            return p_z.sample(sample_shape)

    def training_step(self, batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()

        x = batch[0]

        #modification for conditional vars
        if self.conditional_vars>0:
            x_cond = batch[1]
            encoder_input = torch.cat([x, x_cond], dim=-1)
            encoder_input = encoder_input.flatten(start_dim=1)
        else:
            encoder_input = x.flatten(start_dim=1)

        x = x.flatten(start_dim=1)

        if self.encoder is not None:
            # q_z = self.encoder(x)
            #modification for conditional vars
            q_z = self.encoder(encoder_input)

            z = q_z.rsample()

            #modification for conditional vars
            if self.conditional_vars>0:
                if self.prior_conditioner is None:
                    p_z = D.Independent(
                        D.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)),
                        reinterpreted_batch_ndims=1,
                    )
                else:
                    p_z = self.prior_conditioner(x_cond)
            else:
                p_z = D.Independent(
                    D.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)),
                    reinterpreted_batch_ndims=1,
                )
            assert q_z.log_prob(z).shape == p_z.log_prob(z).shape, "KL terms must have matching batch shapes"
            kl_z = q_z.log_prob(z) - p_z.log_prob(z) # Monte Carlo estimate of KL divergence

            beta = 1.
            if self.regularize and self.trainer.current_epoch < self.warmup:
                beta = min(1., kl_z.mean() / z.shape[1])
        else:
            z = None

        #modification for conditional vars
        if self.conditional_vars>0:
            decoder_input = torch.cat([z, x_cond], dim=-1) if z is not None else x_cond
        else:
            decoder_input = z
        decoder_input = decoder_input.flatten(start_dim=1)
        log_prob_x = self.flow(decoder_input).log_prob(x)

        # log_prob_x = self.flow(z).log_prob(x)
        if self.encoder is not None:
            elbo = log_prob_x - beta * kl_z
            loss = -elbo.mean(dim=0)
        else:
            loss = -log_prob_x.mean(dim=0)

        if getattr(self.trainer, "logger", None) is not None:
            self.log('train_loss', loss.detach(), prog_bar=True)
            self.log('log_prob', log_prob_x.mean(dim=0).detach(), prog_bar=True)
            if self.encoder is not None:
                self.log('kl_z', kl_z.mean(dim=0).detach(), prog_bar=True)
            if self.trainer.lr_scheduler_configs:
                self.log('lr', self.lr_schedulers().get_last_lr()[-1], prog_bar=True)

        self.manual_backward(loss)
        opt.step()

        return loss

    def validation_step(self, batch, batch_idx):
        x = batch[0]

        #modification for conditional vars
        if self.conditional_vars>0:
            x_cond = batch[1]
            encoder_input = torch.cat([x, x_cond], dim=-1)
            encoder_input = encoder_input.flatten(start_dim=1)
        else:
            encoder_input = x.flatten(start_dim=1)

        x = x.flatten(start_dim=1)

        if self.encoder is not None:
            #modification for conditional vars
            # q_z = self.encoder(x)
            q_z = self.encoder(encoder_input)
            z = q_z.rsample()

            # modification for conditional vars
            if self.conditional_vars > 0:
                if self.prior_conditioner is None:
                    p_z = D.Independent(
                        D.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)),
                        reinterpreted_batch_ndims=1,
                    )
                else:
                    p_z = self.prior_conditioner(x_cond)
            else:
                p_z = D.Independent(
                    D.Normal(loc=torch.zeros_like(z), scale=torch.ones_like(z)),
                    reinterpreted_batch_ndims=1,
                )
            assert q_z.log_prob(z).shape == p_z.log_prob(z).shape, "KL terms must have matching batch shapes"
            kl_z = q_z.log_prob(z) - p_z.log_prob(z)
            beta = 1. #always 1 for validation
        else:
            z = None
            kl_z = None
            beta = 0.

        #modification for conditional vars
        if self.conditional_vars>0:
            decoder_input = torch.cat([z, x_cond], dim=-1) if z is not None else x_cond
        else:
            decoder_input = z
        log_prob_x = self.flow(decoder_input).log_prob(x)

        # log_prob_x = self.flow(z).log_prob(x)
        if self.encoder is not None:
            elbo = log_prob_x - beta * kl_z
            loss = -elbo.mean(dim=0)
        else:
            loss = -log_prob_x.mean(dim=0)
        if getattr(self.trainer, "logger", None) is not None:
            self.log('val_loss', loss.detach(), on_step=False, on_epoch=True,  prog_bar=True)
            self.log('val_log_prob', log_prob_x.mean(dim=0).detach(), on_step=False, on_epoch=True, prog_bar=False)
            if self.encoder is not None:
                self.log('val_kl_z', kl_z.mean(dim=0).detach(), on_step=False, on_epoch=True, prog_bar=False)
        return loss

    def sample(self, sample_shape: torch.Size, conditional_samples: torch.Tensor|None = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample from the model.
        :param num_samples: number of samples to generate
        :return: tuple of (samples, latent variables)
        """
        z = self.sample_prior(sample_shape, conditional_samples) if self.encoder is not None else None
        decoder_input = z.clone() if z is not None else None
        if decoder_input is None:
            decoder_input = conditional_samples
        else:
            decoder_input = torch.cat([decoder_input, conditional_samples], dim=-1) if conditional_samples is not None else decoder_input
        decoder_input = decoder_input.flatten(start_dim=1) if decoder_input is not None else None
        if decoder_input is None:
            x = self.flow().sample(sample_shape)
        else:
            x = self.flow(decoder_input).sample()
        return x, z

    def sample_interventional(self,
                              index: torch.LongTensor | int | Sequence[int],
                              value: torch.Tensor | float | Sequence[float],
                              sample_shape: torch.Size = (),
                              conditional_samples: torch.Tensor|None = None
                            ) -> Tuple[torch.Tensor, torch.Tensor]:

        """
        Sample from the model with interventional variables.
        :param index: index of the variable to intervene on
        :param value: value to set the variable to
        :param num_samples: number of samples to generate
        :return: samples with the intervened variables
        """
        z = self.sample_prior(sample_shape, conditional_samples) if self.encoder is not None else None
        decoder_input = z.clone() if z is not None else None
        if decoder_input is None:
            decoder_input = conditional_samples
        else:
            decoder_input = torch.cat([decoder_input, conditional_samples], dim=-1) if conditional_samples is not None else decoder_input

        decoder_input = decoder_input.flatten(start_dim=1) if decoder_input is not None else None
        if decoder_input is None:
            x = self.flow().sample_interventional(index=index, value=value, sample_shape=sample_shape)
        else:
            x = self.flow(decoder_input).sample_interventional(index=index, value=value, )
        return x, z

    def compute_counterfactual(self,
                               factual: torch.Tensor,
                               index: torch.LongTensor | int | Sequence[int],
                               value: torch.Tensor | float | Sequence[float],
                               num_samples: int = 10,
                               conditional_samples: torch.Tensor|None = None
                              ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Compute the counterfactual samples.
        :param factual: factual samples
        :param index: index of the variable to intervene on
        :param value: value to set the variable to
        :param num_samples: number of samples of Z to generate to take the mean
        :return: counterfactual samples and samples of posterior of Z
        """

        #modification for conditional samples
        encoder_input = torch.cat([factual, conditional_samples], dim=-1) if conditional_samples is not None else factual
        encoder_input = encoder_input.flatten(start_dim=1) if encoder_input is not None else None

        if self.encoder is not None:
            z  = self.encoder(encoder_input).sample((num_samples,))
            z = z.mean(dim=0)
        else:
            z = None

        #modification for conditional vars
        decoder_input = z.clone() if z is not None else None
        if decoder_input is None:
            decoder_input = conditional_samples
        else:
            decoder_input = torch.cat([decoder_input, conditional_samples],
                                      dim=-1) if conditional_samples is not None else decoder_input

        decoder_input = decoder_input.flatten(start_dim=1) if decoder_input is not None else None
        if decoder_input is None:
            x = self.flow().sample_interventional(index=index, value=value, sample_shape=num_samples)
        else:
            x = self.flow(decoder_input).compute_counterfactual(factual=factual, index=index, value=value)
        return x, z

    def compute_jacobian(self,
                         x: torch.Tensor|None = None,
                         u: torch.Tensor|None = None,
                         conditional_samples: torch.Tensor|None = None,
                         ) -> torch.Tensor:
        assert x is not None or u is not None, "Either x or u must be provided."
        num_samples = x.shape[0] if x is not None else u.shape[0]
        z = self.sample_prior((num_samples,), conditional_samples) if self.encoder is not None else None

        decoder_input = z.clone() if z is not None else None
        if decoder_input is None:
            decoder_input = conditional_samples
        else:
            decoder_input = torch.cat([decoder_input, conditional_samples], dim=-1) if conditional_samples is not None else None
        n_flow = self.flow(decoder_input)

        with torch.enable_grad():
            if x is not None:  # Compute Jacobian at x
                fn = n_flow.transform
                jac = torch.autograd.functional.jacobian(fn, x.mean(0)).cpu().numpy()
            else:  # Compute Jacobian at u
                fn = n_flow.transform.inv
                jac = torch.autograd.functional.jacobian(fn, u.mean(0)).cpu().numpy()

        return jac




