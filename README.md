docker build -t lightgbm-container .

docker run -d --name Edge-1 -p 5001:5000 lightgbm-container
docker run -d --name Edge-2 -p 5002:5000 lightgbm-container
docker run -d --name Edge-3 -p 5003:5000 lightgbm-container

