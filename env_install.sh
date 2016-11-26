#!/bin/bash
# DO NOT USE SUDO WHEN CALLING THIS SCRIPT

programname=$0

function usage {
  echo "usage: $programname"
}

usage

if [[ $EUID == 0 ]]; then
  echo "ERROR: This script must NOT be run as root! This is so that pip3 does"
       "not try to install into root directories."
  exit 1
fi

if [ $OSTYPE != linux-gnu ]; then
  echo "Warning: script is designed for Linux installation. You must manually"\
       "setup python3 and pip3 on your machine if you are not using Linux."\
       "The rest of the script should work fine otherwise."
fi

PACKAGE_MANAGER=""
if [ "$(command which yum)" ]; then
  echo "Using yum install"
  PACKAGE_MANAGER="yum"
elif [ -n "$(command which apt-get)" ] || [ -n "$(command which apt)" ] ; then
  echo "Using apt-get install"
  PACKAGE_MANAGER="apt-get"
else
  echo "Script only supports yum and apt-get package managers. Please"
  echo "manually install python3 and pip3"
fi

sudo -s <<EOF
if [ $PACKAGE_MANAGER ]; then
  echo "Installing python3.5, dev tools, pip3 using package manager."

  $PACKAGE_MANAGER install python3.5 --upgrade
  $PACKAGE_MANAGER install python3.5-dev libxml2-dev libxslt-dev \
                   python3-pycurl --upgrade
  $PACKAGE_MANAGER install python3-pip --upgrade  
fi
EOF

if [ -a .env ]; then
  echo
  while true ; do
    read -p "Delete the existing environment? [Y/n]" -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]] ; then
      rm -rf .env
      break
    elif [[ $REPLY =~ ^[Nn]$ ]] ; then
      break
    else
      continue
    fi
  done
fi

echo "Updating pip3, installing virtualenv with pip3 and setting up "\
     "environment."


pip3 install virtualenv --upgrade

virtualenv --no-site-packages --distribute .env
source .env/bin/activate
if [ -a pip3_requirements.txt ] ; then
  pip3 install -r pip3_requirements.txt --upgrade
else
  echo "Note: pip3_requirements.txt file does not exist, so pip3 will not" \
       "install any packages."
fi

