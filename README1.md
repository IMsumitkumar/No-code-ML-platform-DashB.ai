# DashB.ai
A low code machine learning and data visualization web based platform for babies

---

![dashViz gif](https://github.com/IMsumitkumar/AutoML-DashB.ai/blob/main/images/20201024_140312.gif)

---

![upload_csv](https://github.com/IMsumitkumar/AutoML-DashB.ai/blob/main/images/4.png)
upload data in csv format

---

![data_from_database](https://github.com/IMsumitkumar/AutoML-DashB.ai/blob/main/images/5.png)
fetch data from mysql live database

---

![main_dashboard](https://github.com/IMsumitkumar/AutoML-DashB.ai/blob/main/images/1.png)
main dashboard

---

![data_operation_board](https://github.com/IMsumitkumar/AutoML-DashB.ai/blob/main/images/2.png)
data operation board

---

![dAshViz_datavisualization](https://github.com/IMsumitkumar/AutoML-DashB.ai/blob/main/images/3.png)
by dash-by-plotly
data visualization tool



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
