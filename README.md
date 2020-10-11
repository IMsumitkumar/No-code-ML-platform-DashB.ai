# DashB.ai
A low code machine learning and data visualization web based platform for babies


---
## Create a virtual environmment

     conda create -n <env_name> python=3.x
     conda activate <env_name>
  
---
## Install required pckages

  Clone or download the repo in your local machine -> enter in the directory where  `manage.py` exists and then run
  
     pip install -r requirements.txt
    
## create superuser and migrate the database
   
   python3 manage.py makemigrations
   python3 manage.py migrate
   
   python3 manage.py createsuperuser
      
      enter crendentials
      
      [username:****]
      [email:*******]
      [password:****]
      
## for email recovery (if u want), you have to set some credentials 
    
    HEAD to DashB --> settings.py -->
      
      set your email and passwoord
