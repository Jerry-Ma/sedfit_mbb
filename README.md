# Zhiyuan's SED fitting

## Installation

1. unpack the tarball to any directory

```
$ cd /your/directory/to/hold/the/code
$ tar -zxvf /path/to/the/source/code/tarball
```

2. install python dependency packages

```
$ pip3 install -r requirements.txt
```

3. edit sedfit_example.py, to change the import path of the package:

```
  ... near the bottom of the file ...
  import sys
> sys.path.insert(0, '/home/ma/Codes/pyjerry')
  from sedfit import core
```

   then change the string `/home/ma/Codes/pyjerry` in the marked line to
   the path on your machine which holds the `sedfit` directory

4. test installation by running the example [Run the example]

## Run the example

    $ cd example                         # go to example directory
    $ python do_fit.py                   # run the fitting
    $ python do_fit.py all               # plot the result for all objects
    $ python do_fit.py obj_1             # plot the result for obj_1
    $ python do_fit.py obj_1,obj_2 True  # plot the result for obj_1 and
                                         # obj_2, and save the figure as
                                         # an eps file.

## Create and run your own fitting job

1. create a copy of `sedfit_example.py` to the directory that contains
   the input catalog. Note that the filename can be anything (here I used
   `do_fit.py`), as long as it ends with `.py`

```
$ cp sedfit_example.py "/working/directory/do_fit.py"
```

2. edit do_fit.py by following the comments in the file. The purpose of
   doing so is to instruct the program about where to find the input
   catalog and what's the content of it.

3. run the fitting (refer to [Run the example] section)
