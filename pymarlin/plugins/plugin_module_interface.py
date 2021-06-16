'''plugin module interface'''
import os
from transformers import AutoTokenizer, AutoConfig
from pymarlin.core import module_interface, data_interface

class PluginModuleInterface(module_interface.ModuleInterface):
    '''Common plugin module interface to easily load Huggingface tokenizers and Configs'''
    def auto_setup(self, automodel_class):
        """Run all (tokenizer,config,model) setups"""
        self.setup_tokenizer()
        self.setup_model_config()
        self.setup_model(automodel_class)

    @property
    def data(self):
        """DataInterface object that is used to retrieve corresponding train or val dataset.

        Returns:
            data: DataInterface object with at least one of train or val data.
        """
        return self._data

    @data.setter
    def data(self, datainterface):
        assert isinstance(datainterface, data_interface.DataInterface)
        assert (
            len(datainterface.get_train_dataset()) != 0
            or len(datainterface.get_val_dataset()) != 0
        )
        self._data = datainterface

    @property
    def model(self):
        """Pytorch model."""
        return self._model

    @model.setter
    def model(self, newmodel):
        self._model = newmodel

    def setup_tokenizer(self):
        """Initializes AutoTokenizer from
        model_args.tokenizer_path or model_args.hf_model string
        """
        if self.args.model_args.tokenizer_path is not None:
            tokenizer = AutoTokenizer.from_pretrained(
                self.args.model_args.tokenizer_path
            )
        else:
            tokenizer = AutoTokenizer.from_pretrained(self.args.model_args.hf_model)

        self.tokenizer = tokenizer

    def setup_model_config(self):
        """Initializes AutoConfig from
        model_args.model_config + model_args.model_config_file path or model_args.hf_model string
        """
        if self.args.model_args.model_config_path is not None:
            model_config = AutoConfig.from_pretrained(
                os.path.join(
                    self.args.model_args.model_config_path,
                    self.args.model_args.model_config_file,
                )
            )
        else:
            model_config = AutoConfig.from_pretrained(
                self.args.model_args.hf_model
            )

        model_config.num_labels = (
            len(self.data.get_labels()) if hasattr(self.data, "get_labels") else None
        )
        self.model_config = model_config

    def setup_model(self, automodel_class):
        """Initializes automodel_class arg by either:
        Option 1: Load weights from specified files mentioned in YAML config
            model:
                model_config_path
                model_config_file
                model_path
                model_file
        Option 2: Load from Huggingface model hub, specify string in YAML config as:
            model:
                hf_model

        Args:
            automodel_class: Huggingface AutoModelFor* class
        """
        if (
            self.args.model_args.model_path is not None
            and self.args.model_args.model_file is not None
        ):
            self.model = automodel_class.from_pretrained(
                os.path.join(
                    self.args.model_args.model_path, self.args.model_args.model_file
                ),
                config=self.model_config,
            )
        else:
            self.model = automodel_class.from_pretrained(
                self.args.model_args.hf_model, config=self.model_config
            )
