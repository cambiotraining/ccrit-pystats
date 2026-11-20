# CCRIT Python statistics

`ccrit-pystats` provides Python helpers for CCRIT statistics courses. The first
helper is the diagnostic panel from the
[Core Statistics course](https://cambiotraining.github.io/corestats/).

## Install

From a copy of this repository, run this command in the Python environment
you use for the course.

```sh
python -m pip install .
```

## Plot diagnostics

```python
import pandas as pd
import statsmodels.formula.api as smf

from ccrit_pystats import dgplots

data = pd.read_csv("data/CS3-usarrests.csv")
fit = smf.ols("murder ~ assault", data=data).fit()
dgplots(fit)
```

The function returns a Matplotlib figure. With the inline notebook backend,
put the call at the end of a cell to display it once. If you assign the figure
with `fig = dgplots(fit)`, use `display(fig)` to display it. In a script,
call `matplotlib.pyplot.show()` to display it or use
`fig.savefig("diagnostics.png")` to save it.

The figure shows residuals versus fitted values, a normal Q–Q plot,
scale–location, and Cook's distance. It also works with an ANOVA fitted using
`smf.ols()`.

## Build a release

See the [changelog](CHANGELOG.md) for the package's release history.

```sh
python -m pip install build
python -m build --wheel
```

Attach the wheel in `dist/` to a tagged GitHub release and put its direct URL
in the course setup instructions as `python -m pip install <wheel-url>`.
Learners can install that version using pip without setting up Git.

## Run the tests

```sh
python -m pip install .
python -m unittest discover -s tests
```

Use British English for documentation, comments and user-facing text.
