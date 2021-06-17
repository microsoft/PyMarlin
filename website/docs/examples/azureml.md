---
title: Azure ML
---

You can use [`examples/azureml/azureml_submit.py`](https://github.com/microsoft/PyMarlin/blob/main/examples/azureml/azureml_submit.py)
to submit examples to run on Azure ML.

For example:

```bash
python examples/azureml/azureml_submit.py --marlin_example classification --backend ddp-amp --process_count 2
```

See `examples/azureml/azureml_submit.py -h` for more options.

**Note.** please modify the ['examples/azureml/config.json'](https://github.com/microsoft/PyMarlin/blob/main/examples/azureml/config.json)
to reference your desired Azure ML workspace. See [Azure ML CheatSheet](https://aka.ms/aml/cheatsheet) for more details.
