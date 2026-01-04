

conda env create -f morl_base_d.yaml


sudo add-apt-repository ppa:sumo/stable
sudo apt-get update
sudo apt-get install sumo sumo-tools sumo-doc

echo 'export SUMO_HOME="/usr/share/sumo"' >> ~/.bashrc
source ~/.bashrc

pip install torch==2.1.0

conda activate morl_base_d


pip install --upgrade pip
pip install swig
pip install gymnasium[box2d]
pip install scikit-learn


<Run>
1) Traffic
python envelope_big_intersection.py -env -rd 4 -dt 20 -nevw 20 -efq 52 -mlr 0.0003 -hodec 39000 -evf 'equal' -base 'ours' -rlr 0.0003 -rint 5 -drp 0.75 -se 0 

2) LunarLander
python envelope_lunar_lander.py -env -rd 3 -nevep 1 -nevw 15 -efq 400000 -mlr 0.0003 -hodec 500000 -evf 'equal' -base 'ours' -rlr 0.0003 -rint 5 -drp 0.25 -se 0
