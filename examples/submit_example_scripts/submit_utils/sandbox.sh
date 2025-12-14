#!/bin/sh
# list information about "sandbox" or private job area on execution point
echo 'Date: ' `date`
echo 'Host: ' `hostname` 
echo 'Sandbox: ' `pwd` 
ls -alF
# END
