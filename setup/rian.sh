REMOTE_USER="rian"
REMOTE_HOST="192.168.50.111"
REMOTE_PASSWORD="rian123"

echo "CONNECTING TO RIAN'S VEHICLE"

sshpass -p "$REMOTE_PASSWORD" ssh -t -o StrictHostKeyChecking=no "$REMOTE_USER@$REMOTE_HOST" \
    "bash --rcfile <(echo 'source project/slam_ws/install/setup.bash; echo \"Livox sourced!\"; echo \"ROS_DISTRO: \$ROS_DISTRO\"')"
