pipeline  {
    agent any

    stages{
        stage("Clone Repository") {
            steps{
                script{
                    echo "Cloning Repository to Jenkins Workspace"
                    checkout scmGit(branches: [[name: '*/main']], extensions: [], userRemoteConfigs: [[credentialsId: 'github-token', url: 'https://github.com/DMC2003version1/hotel_project_mlops.git']])
                }
            }
        }
    }
}