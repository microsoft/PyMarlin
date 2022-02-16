# Differentially Private training in PyMarlin

This page provides a brief overview of how to train models with Differential Privacy in PyMarlin.
We utilize the [Opacus package v1.0](https://opacus.ai/) for the implementation of DP training.
We refer the reader to Appendix A-B for a brief introduction into Differential Privacy (DP) and Differentially Private Stochastic Gradient Decent (DP-SGD).

## Configuration for DP training in PyMarlin
- **TrainerBackend Name**: Training backend has special configurations for DP training.
Namely, “sp-dp” is used for single process DP training and “ddp-dp” is used for distributed DP training.
- **Global clipping**: DP clips per sample gradients, therefore, the global clipping in PyMarlin (“clip_grads” argument) should be set to “False”.
Otherwise, an error will be raised to highlight this point.
- **DifferentialPrivacyArguments**: DP training expects “differential_privacy_args” in TrainerBackendArguments.
A dataclass named DifferentialPrivacyArguments is introduced in differential_privacy.py script inside the utils folder.
There are four parameters to be passed regarding DP:
    - **noise_multiplier**: This parameter is the scaling for the noise variance.
    The higher this parameter is the better the privacy budget will be (i.e., smaller epsilon), however, the utility degrades in return.
    Typically, this parameter would be around 1.0 (it might differ for different scenarios).
    - **per_sample_max_grad_norm**: This parameter clips the per sample gradients.
    It has been observed that small values around (0.1-1) work well in practice and the performance does not depend much on this parameter unless picked somewhere high.
    - **sample_rate**: For sample-level DP, let effective batch size be <img src="https://render.githubusercontent.com/render/math?math=B"> and the total number of samples in the training set be <img src="https://render.githubusercontent.com/render/math?math=N">.
    This should be set as <img src="https://render.githubusercontent.com/render/math?math=B/N">.
    For user-level DP and beyond, please see the related section below.
    - **delta**:
    This parameter represents the probability of failure for DP guarantees, typically set as <img src="https://render.githubusercontent.com/render/math?math=\ll N^{-1}"> where <img src="https://render.githubusercontent.com/render/math?math=N"> is the number of samples in the training set.
    This parameter is not required for training, only used to calculate the privacy budget (i.e., epsilon).

## Tips for hyperparameter selection 

- **Batch size**: It has been observed that large batch sizes help achieving good privacy/utility trade-off in DP training.
Choosing batch size at least around square root of the number of samples in the training set can be considered as a rule of thumb.
DP training requires larger memory allocation, therefore, a large batch size can be achieved effectively using gradient accumulation.
PyMarlin has an argument “gpu_batch_size_limit” that one can use to limit the batch size per GPU and the training will handle the gradient accumulation itself.
As a reference point, we can fit HuggingFace GPT-2 model with sequence length 128 and batch size 32 on a 32GB GPU.
Torch version may sometimes cause issues with memory allocation; we used torch version 1.9.1 and transformers version 4.10.3 for this reference. 
- **Learning rate**: This parameter would benefit hyperparameter search.
Constant learning rate with AdamW optimizer has provided good performance in various setups.
However, other optimizers & LR schedulers have not been extensively explored yet. 

## Calculating the privacy budget (i.e., epsilon) 

In DP training, there is an accountant connected to the “trainer_backend” that can be used to calculate the privacy budget.
There is some dependency based on backend being “sp-dp” or “ddp-dp”, the example script below is an illustration of how to calculate the privacy budget (i.e., epsilon).
```
if config["tr"]["backend"] == 'ddp-dp' or config["tr"]["backend"] == 'sp-dp':
    if config['tr']['backend'] == 'ddp-dp':
        accountant = trainer.trainer_backend.trainer_backend.accountant
    else:
        accountant = trainer.trainer_backend.accountant

    eps_rdp = accountant.get_epsilon(delta=1.0/len(data.train_ds))
    print(f"final_epsilon_rdp: {eps_rdp}")
```

## Picking the “noise_multiplier” 

The final privacy budget (i.e., epsilon) depends on “noise_multiplier”, “sample_rate”, "number of optimization steps (i.e., number of steps where the gradient update occurs)”, and “delta”. 
Opacus package has [get_noise_multiplier](https://github.com/pytorch/opacus/blob/ce7b6464688227818595fce7851b96cfec55de00/opacus/accountants/utils.py#L29) method that can be used to calculate the “noise_multiplier” by fixing the other parameters. 
Microsoft has released an [accountant](https://github.com/microsoft/prv_accountant) that is more effective than the privacy accountant Opacus package uses. 
We suggest this accountant to achieve a tighter privacy budget (i.e., epsilon) under the same set of parameters.

*Internal only: We provide [find_noise_multiplier](https://dev.azure.com/ii-m365/PPML/_git/opacus-utils?path=/src/opacus_utils/dp_utils.py) method based on the [PRV accountant](https://github.com/microsoft/prv_accountant).*

## Module incompatibility issues with Opacus 

DP training requires calculation of per sample gradients and Opacus package handles this by adding forward/backward hooks.
Opacus works module-level and the package includes per sample gradient calculation for the most common modules.
If your module has these basic modules in its forward function, you should be good to go.

*Internal only: See our [repo](https://dev.azure.com/ii-m365/PPML/_git/opacus-utils) for HuggingFace GPT-2 series where we register the per sample gradient calculation for certain modules (e.g., Conv1D).*

## “sample_rate” for user-level DP and beyond 

DP training can be performed to protect any entity: sample, user, group, tenant etc.
It all depends on how we form a batch during training.
Let <img src="https://render.githubusercontent.com/render/math?math=N"> be the number of entities in the training set.
Basically, we form a batch of size <img src="https://render.githubusercontent.com/render/math?math=B"> by going over <img src="https://render.githubusercontent.com/render/math?math=B"> entities and obtaining one data sample from each entity.
This means that one epoch will go over <img src="https://render.githubusercontent.com/render/math?math=N"> entities and each entity will be seen only at once during the epoch and there will be <img src="https://render.githubusercontent.com/render/math?math=N/B"> optimization steps.
The data sample taken from each entity across different epochs do not matter (could be the same sample repeatedly, the next sample from its queue, a random sample etc.).
The “sample_rate” will continue to be <img src="https://render.githubusercontent.com/render/math?math=B/N"> and it will satisfy entity-level DP.
Note that <img src="https://render.githubusercontent.com/render/math?math=N"> decreases as the entity becomes larger (e.g., number of samples >> number of tenants), therefore, privacy/utility trade-off becomes more challenging accordingly.

We note that the procedure above provides a “global” model training, treating each entity equivalently even if the data distributions differ substantially (some users having ~1k samples whereas others having only ~10).
Another way to achieve entity-level DP is to cap the number of contributions per entity by some constant <img src="https://render.githubusercontent.com/render/math?math=K"> and continue the training just like sample-level DP.
In this case, the total number of samples in the training set will depend on both the number of entities and the cap <img src="https://render.githubusercontent.com/render/math?math=K">; let it be <img src="https://render.githubusercontent.com/render/math?math=N"> and batch size be <img src="https://render.githubusercontent.com/render/math?math=B">.
The “sample_rate” in this case will be equal to <img src="https://render.githubusercontent.com/render/math?math=K*B/N"> and this procedure will also provide entity-level DP.
Note that here entities that have contributions close to <img src="https://render.githubusercontent.com/render/math?math=K"> might enjoy better performance than the ones that have contributions <img src="https://render.githubusercontent.com/render/math?math=\ll K">.
On the other hand, privacy is calculated as a worst-case guarantee, therefore, we will calculate the privacy budget with “sample_rate” as <img src="https://render.githubusercontent.com/render/math?math=K*B/N"> although entities that have contributions <img src="https://render.githubusercontent.com/render/math?math=\ll K"> will enjoy better privacy guarantees essentially. 

*Internal only: For a more formal treatment, see the following [document](https://microsoft-my.sharepoint.com/:b:/p/sigopi/EVDYL97UflFIgq2VlvC414UBhj3MBBSm1n40EcCE78PByw?e=tHnbrH).* 

## Using more than one optimizer in DP training 

We expect that there will be a single optimizer that updates the model parameters with DP-SGD.
On a very rare occasion that one has additional optimizer(s) that update(s) disjoint set of model parameters without DP, then the user must specify which optimizer(s) will be used without DP.
This can be done easily by wrapping such optimizers with NoDPWrap class introduced in differential_privacy.py script inside the utils folder.

### Appendix A - Introduction to Differential Privacy (DP)

Differential Privacy is a property of a stochastic algorithm <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}"> (in our case SGD).
In short, it quantifies how similar the distributions of the output of the algorithm <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}"> are when they are trained on any two adjacent datasets <img src="https://render.githubusercontent.com/render/math?math=D"> and <img src="https://render.githubusercontent.com/render/math?math=\bar{D}">.
Two datasets <img src="https://render.githubusercontent.com/render/math?math=D"> and <img src="https://render.githubusercontent.com/render/math?math=\bar{D}"> are defined to be adjacent if they differ in not more than one participant's data.

If this is the case, we call <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}(D)"> and <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}(\bar{D})"> that <img src="https://render.githubusercontent.com/render/math?math=(\varepsilon, \delta)">-indistinguishable and write <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}(D) \approx_{\varepsilon, \delta} \mathcal{A}(\bar{D})"> where <img src="https://render.githubusercontent.com/render/math?math=\varepsilon > 0"> and <img src="https://render.githubusercontent.com/render/math?math=\delta \in [0,1]">.
In general the smaller <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and <img src="https://render.githubusercontent.com/render/math?math=\delta"> are the more similar the distributions are.

This is a nice concept for privacy since it gives rise to the notion of plausible deniability.
Let's assume a participant is unsure about their data being part of a dataset <img src="https://render.githubusercontent.com/render/math?math=D">.
In this case, they can be assured as there always is a dataset <img src="https://render.githubusercontent.com/render/math?math=\bar{D}"> to which they didn't contribute and training algorithm would have produced a very similar result.

This notion of plausible deniability is contingent on a suitable choice of <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> and <img src="https://render.githubusercontent.com/render/math?math=\delta">.
In order to interpret these variables better, let's look at the definition of <img src="https://render.githubusercontent.com/render/math?math=(\varepsilon, \delta)">-indistinguishability.
We can define <img src="https://render.githubusercontent.com/render/math?math=(\varepsilon, \delta)">-indistinguishability as the following.
For any subset <img src="https://render.githubusercontent.com/render/math?math=S \subseteq O"> where <img src="https://render.githubusercontent.com/render/math?math=O"> is the set of all possible outputs of <img src="https://render.githubusercontent.com/render/math?math=\mathcal{A}"> we require  
  
<img src="https://render.githubusercontent.com/render/math?math=\text{Pr}(\mathcal{A}(D) \in S) \leq \text{e}^{\varepsilon} \text{Pr}(\mathcal{A}(\bar{D}) \in S) %2b \delta">  
  
Here <img src="https://render.githubusercontent.com/render/math?math=\delta"> represents the probability of failure.
We therefore typically require that <img src="https://render.githubusercontent.com/render/math?math=\delta \ll |D|^{-1}">.
Choosing a suitable <img src="https://render.githubusercontent.com/render/math?math=\varepsilon"> is harder and should be chosen on a case by case basis.
As a rough rule of thumb we can consider <img src="https://render.githubusercontent.com/render/math?math=\varepsilon < 4"> to provide adequate privacy for deep neural network training.

### Appendix B - Introduction to Differentially Private Stochastic Gradient Decent (DP-SGD)

In order to make deep neural network training Differentially Private, two modifications to the standard Stochastic Gradient Decent algorithm are required.

1. Sample gradients need to be clipped to a hyper-parameter <img src="https://render.githubusercontent.com/render/math?math=C">
2. Batch gradients are noised with <img src="https://render.githubusercontent.com/render/math?math=C \sigma \mathcal{N}(0,1)">

The resulting algorithm is often referred to [DP-SGD](https://arxiv.org/pdf/1607.00133.pdf).
However, the same modifications make Adam, Adagrad, etc. differentially private as well.

Several packages implement DP-SGD and make switching to Differentially Private training relatively easy.
However, it should be noted that the sample gradient clipping renders a lot of the parallelization gained by using batch operations ineffective.
Consequently, DP-SGD training is often slower than regular training.
Finding more performant DP-SGD algorithms is a field of ongoing research and several promising approaches have been developed recently.
 





 





