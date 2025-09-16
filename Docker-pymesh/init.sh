#!/bin/bash
set -e
USER_ID=${LOCAL_UID:-9001}
GROUP_ID=${LOCAL_GID:-9001}

echo "Starting__ with UID: $USER_ID, GID: $GROUP_ID"
#usermod -u $USER_ID sem
#groupmod -g $GROUP_ID sem
#export HOME=/home/sem
#id sem
exec /bin/bash
echo /bin/bash

