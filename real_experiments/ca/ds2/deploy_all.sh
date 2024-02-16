
# /bin/bash

# cilantro test
kubectl apply -f auth_default_user.yaml
kubectl apply -f coding-assignment-deployment.yaml
kubectl apply -f coding-assignment-service.yaml
sleep 2
kubectl apply -f config_scheduler_cilantro.yaml
kubectl apply -f metric-server.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
sleep 5
kubectl apply -f ingress.yaml
kubectl apply -f ca-client.yaml
sleep 3660
./fetch_results.sh
sleep 10
./clean_kind_cluster.sh
sleep 10

# ds2 test
kubectl apply -f auth_default_user.yaml
kubectl apply -f coding-assignment-deployment.yaml
kubectl apply -f coding-assignment-service.yaml
sleep 2
kubectl apply -f config_scheduler_ds2.yaml
kubectl apply -f metric-server.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
sleep 5
kubectl apply -f ingress.yaml
kubectl apply -f ca-client.yaml
sleep 3660
./fetch_results.sh
sleep 10
./clean_kind_cluster.sh
sleep 10

# k8s test
kubectl apply -f auth_default_user.yaml
kubectl apply -f coding-assignment-deployment.yaml
kubectl apply -f coding-assignment-service.yaml
sleep 2
kubectl apply -f config_scheduler_k8s.yaml
kubectl apply -f metric-server.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
sleep 5
kubectl apply -f ingress.yaml
kubectl apply -f ca-client.yaml
sleep 3660
./fetch_results.sh
sleep 10
./clean_kind_cluster.sh
sleep 10

# pid test
kubectl apply -f auth_default_user.yaml
kubectl apply -f coding-assignment-deployment.yaml
kubectl apply -f coding-assignment-service.yaml
sleep 2
kubectl apply -f config_scheduler_pid.yaml
kubectl apply -f metric-server.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
sleep 5
kubectl apply -f ingress.yaml
kubectl apply -f ca-client.yaml
sleep 3660
./fetch_results.sh
sleep 10
./clean_kind_cluster.sh
sleep 10

# jingle test
kubectl apply -f auth_default_user.yaml
kubectl apply -f coding-assignment-deployment.yaml
kubectl apply -f coding-assignment-service.yaml
sleep 2
kubectl apply -f config_scheduler_jingle.yaml
kubectl apply -f metric-server.yaml
kubectl apply -f https://raw.githubusercontent.com/kubernetes/ingress-nginx/main/deploy/static/provider/kind/deploy.yaml
sleep 5
kubectl apply -f ingress.yaml
kubectl apply -f ca-client.yaml
sleep 3660
./fetch_results.sh
sleep 10
./clean_kind_cluster.sh
sleep 10
