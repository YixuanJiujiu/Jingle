
# /bin/bash
#kubectl apply -f .

kubectl apply -f auth_default_user.yaml
kubectl apply -f web-serving-deployment.yaml
kubectl apply -f web-serving-service.yaml
sleep 2
kubectl apply -f config_scheduler_jingle.yaml
kubectl apply -f metric-server.yaml

kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml

sleep 10

kubectl apply -f ingress.yaml

kubectl apply -f ca-client.yaml

sleep 3660

./fetch_results.sh


sleep 10
