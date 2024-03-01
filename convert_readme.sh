# I personally do not like the way the github markdown format views, but I like Jupyter notebooks.
# Thanks to https://andrewpwheeler.com/2021/09/06/using-jupyter-notebooks-to-make-nice-readmes-for-github,
# I can convert my README.ipynb to a README.md file and avoid working on README.md directly.

jupyter nbconvert --execute --to markdown README.ipynb
