# Run on machines/testbeds with no root privs in ${MINIGAN_ROOT}
 
PIP=pip3
PYT=python3

#${PIP} install --upgrade pip --user

${PIP} install --user virtualenv

${PYT} -m venv minigan_torch_env

source minigan_torch_env/bin/activate

${PIP} install numpy 
${PIP} install torch
${PIP} install horovod
${PIP} install tensorboard
${PIP} install matplotlib==3.0.0

${PIP} list


