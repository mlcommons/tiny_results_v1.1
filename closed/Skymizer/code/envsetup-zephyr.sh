sudo apt update
sudo apt -y upgrade
sudo apt -y install git libtinfo5 python3 python3-pip python3-venv mercurial wget
# kws
sudo apt -y install ffmpeg
pip3 install wheel==0.38.4
# zephyr
wget https://apt.kitware.com/kitware-archive.sh
sudo bash kitware-archive.sh
sudo DEBIAN_FRONTEND=noninteractive apt install -y \
  --no-install-recommends git cmake ninja-build gperf \
  ccache dfu-util device-tree-compiler wget \
  python3-dev python3-pip python3-setuptools python3-tk python3-wheel xz-utils file \
  make gcc gcc-multilib g++-multilib libsdl2-dev
sudo apt -y install stlink-tools
sudo systemctl restart udev
