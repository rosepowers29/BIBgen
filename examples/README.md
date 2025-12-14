# Preprocessing and Readying Data

## File Conversion
Run ``python convert_slcio_files.py -i [inputfiles] -o [outputfile]`` on a machine with pyLCIO configured. ``inputfiles`` should be LCIO format, ``outputfile`` should end with ``.hdf5``.

## Make Training Data
Run the ``make_training_data.py`` script. See usage notes in file.

## Make Diffused Data
Run the ``diffused_data.py`` script with commandline inputs as defined in the script.

## Now you are ready to train!

For the rest of the demo, follow the instructions in ``submit_example_scripts/README.md`` and use the demo scripts in ``generation``.

## Analysis

To plot your data, run ``python plot_comparison [mc_file] [gen_file] -o [output filename]``
