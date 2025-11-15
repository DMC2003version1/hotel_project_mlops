FROM jenkins/jenkins:lts

USER root

RUN apt-get update -y && \
    apt-get install -y ca-certificates curl gnupg lsb-release && \
    install -m 0755 -d /etc/apt/keyrings && \
    curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc && \
    chmod a+r /etc/apt/keyrings/docker.asc && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian $(. /etc/os-release && echo $VERSION_CODENAME) stable" > /etc/apt/sources.list.d/docker.list && \
    apt-get update -y && \
    apt-get install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

RUN groupadd -f docker && usermod -aG docker jenkins

RUN mkdir -p /var/lib/docker
VOLUME /var/lib/docker

# FIX MISSING PLUGINS HERE
RUN jenkins-plugin-cli --plugins \
    gson-api \
    plugin-util-api \
    workflow-support \
    checks-api \
    trilead-api

USER jenkins
