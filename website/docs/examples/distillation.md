# Distillation

With `pymarlin` library, distillation can be done in a standalone manner or as an extension to your original training Scenario. In this example, we will go through how the [GLUE Task](glue-tasks.md) setup was extended to also perform distillation.

Data Preprocessing is the same as [here](glue-tasks.md). The main implementation is in the `ModuleInterface` which we chose to call `DistillRecipe` (inheriting from the GLUE `Recipe`).

The key methods of `DistillRecipe` that we want to override:
1. Setting up teacher and student model and related items such as config as needed. Here, we have the option to modify the student config depending on the desired changes to the depth or width of the model.
    ```python
        def setup_models(self):
            self._setup_configs()
            # teacher setup
            self.teacher = AutoModelForSequenceClassification.from_pretrained(
                os.path.join(self.args.model_args.model_wts_path, self.args.model_args.model_file),
                config=self.model_config
                )
            # student setup
            self.model = copy.deepcopy(self.teacher)
            if len(self.student_layers) > 0:
                layer_modules = getattr(self.model, self.args.model_args.encoder_key).encoder.layer
                new_layer_modules = distill_utils.extract_layers(layer_modules, self.student_layers)
                getattr(self.model, self.args.model_args.encoder_key).encoder.layer = new_layer_modules
     
            self.teacher.eval()
            self.output_hidden = True if 'hidden_states' in self.loss_types else False
            self.output_attentions = True if 'attentions' in self.loss_types else False
            return (self.model, self.teacher)
    ```

2. Modify `train_step` to set teacher in eval mode, get teacher outputs, get student outputs, and compute a custom loss. The loss can be a combination of `logits`, `labels` or various intermediate representations such as `hidden_states` and `attentions`. You have the flexibility to determine your distillation logic.
    ```python
        def train_step(self, global_step, batch, device):
            self.teacher.eval()
            inputs = self._inputs_to_device(batch, device)
            teacher_outputs = self.teacher.forward(**inputs,
                                output_hidden_states=self.output_hidden,
                                output_attentions=self.output_attentions,
                                ) # label_loss, logits, hidden, attns
            student_outputs = self.model.forward(**inputs,
                                output_hidden_states=self.output_hidden,
                                output_attentions=self.output_attentions,
                                )
            total_loss = torch.zeros([1], dtype=student_outputs[0].dtype, device=device)
            for i, k in enumerate(self.loss_types):
                if k == 'labels':
                    student_scores = student_outputs.loss
                    teacher_scores = teacher_outputs.loss
                else:
                    student_scores = getattr(student_outputs, k)
                    teacher_scores = getattr(teacher_outputs, k)

                if student_scores is not None and teacher_scores is not None:
                    if k == 'logits':
                        total_loss += self.loss_weights[i] * distill_utils.logits_loss(
                            student_scores, teacher_scores,
                            temperature=self.distill_args.temperature,
                        )
                    elif k != 'logits' and self.distill_args.width_shrinkage == 0:
                        total_loss += self.loss_weights[i] * distill_utils.representations_loss(
                                        student_scores,
                                        teacher_scores,
                                        [*range(len(self.student_layers))],
                                        self.student_layers
                        )
            return total_loss
    ```

3. As an example, `on_end_train` can be used to cleanup any changes made to the final student model config and save it to the output directory along with the student model.

That's it! If you have a scenario setup it's as easy as overriding just 2 methods. 