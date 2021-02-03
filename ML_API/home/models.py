from django.db import models

# Create your models here.

class inp(models.Model):
  url = models.CharField(max_length = 500,null = True)