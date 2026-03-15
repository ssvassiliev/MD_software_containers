#!/bin/bash

if [ $# -ne 3 ] 
then
	echo 'Usage: $0 <nGPUs> <jobscript> <nreps>'
	exit 1
fi

for i in `seq 1 ${3}`
do 
    workdir=_${1}_${i}_`basename -s .sh ${2}`
    rm -rf ${workdir}
    mkdir ${workdir} && cd ${workdir} 
    ln -s ../prmtop.parm7 ./prmtop.parm7
    ln -s ../restart.rst7 ./restart.rst7 
    ln -s ../openmm_input_rocm.py ./openmm_input_rocm.py
    bash  ../${2} ${1}
    cd ..
done

