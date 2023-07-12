cd "$(dirname "${BASH_SOURCE[-1]}")"
curl -J https://uni-bonn.sciebo.de/s/E07eq2AVzOrKORr/download -o ../data/robocup_bordeaux_23.zip
unzip ../data/robocup_bordeaux_23.zip -d ../data
rm ../data/robocup_bordeaux_23.zip
