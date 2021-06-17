---
title: Azure ML
---

You can use [`examples/classification/azureml/azureml_submit.py`](https://github.com/microsoft/PyMarlin/blob/main/examples/classification/azureml/azureml_submit.py)
to submit examples to run on Azure ML.

For example:

```bash
cd examples/classification/azureml/
python azureml_submit.py --backend ddp-amp --process_count 2
```

See `examples/classification/azureml/azureml_submit.py -h` for more options.

**Note.** please add the config.json for to reference your desired Azure ML workspace. See [Azure ML CheatSheet](https://aka.ms/aml/cheatsheet) for more details.
