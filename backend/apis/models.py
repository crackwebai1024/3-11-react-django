from django.db import models

# Create your models here.
class Sentiment(models.Model):
    title = models.CharField(max_length=50)
    teacher = models.CharField(max_length=50)

    def __str__(self):
        return self.title

class History(models.Model):
    token = models.CharField(max_length=100)
    action = models.CharField(max_length=50)
    text = models.TextField(max_length=5000)
    time = models.DateField()
    
    def __str__(self):
        data = self.action +"{{{{{***}}}}}"+ self.text + "{{{{{***}}}}}" + str(self.time)
        return data 
