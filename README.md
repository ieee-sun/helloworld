### hello world!
Testing the first repo

---

### Q - should I put `.py` or only Jupyter notebooks (`.ipynb`) in Github?
##### by [Melonface](https://www.reddit.com/user/MelonFace/): 
> how I'd say you do it. I work in the industy.
> 1. Develop in jupyter / google colab.
>
> 2. Once it runs, find things or patterns you're doing many times, make functions for those in a `utils.py` or similar collection of tools you may reuse in the future.
>
> 3. Take any non-general functions in the notebook and move them to a `my_project_stuff.py` (obviously better name).
>     * `%load_ext autoreload native jupyter` plugin will let you change these functions without restarting the kernel.
>
> 5. Parameterize and move the model definition into `name_model.py`
> 6. Now your notebook should contain only the data loading and training code. This is fine. And still gives you the debug/experiment freedom.
>
> 7. Personally I also put the training code in a `.py` file at the end and load checkpoints into notebooks as needed to debug or fine tune. But that's personal taste. The reason for this is the service I use to run training in cloud. Where it's more reliable to run long running python programs rather than jupyter kernels.

##### by [ktpr](https://www.reddit.com/user/ktpr/):
> You could convert it to a [CLI integration](https://docs.github.com/en/github-cli/github-cli/quickstart) , a cooperative layout on Github -
> ```
>    pip install -U pipenv cookiecutter
>    cookiecutter gh:crmne/cookiecutter-modern-datascience
> ```
> Copy your `.ipynb` notebooks into `./notebook/`
> - Step 1: from your notebooks, place data, pipeline and model code into their respective directories. Wrap with Typer for CLI integration
> - Step 2: in `./serve`, make an `app.py` file that calls appropriate code to expose your ML project as an API. 
> - Endpoints can load, train, predict and otherwise serve your project.




---
> ###### Prepared by:
> ###### Sun CHUNG, *SMIEEE* M.Sc. HKU

---
###### End of File.


