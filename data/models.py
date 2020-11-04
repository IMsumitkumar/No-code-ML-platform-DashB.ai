from django.db import models
from django.contrib.auth.models import User

# Create your models here.
class DataSet(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='files', blank=True, null=True)

    def __str__(self):
        return str(self.user)


class ProcessedDataSet(models.Model):
    user = models.ForeignKey(User, on_delete=models.CASCADE)
    file = models.FileField(upload_to='saved', blank=True, null=True)

    def __str__(self):
        return str(self.file)