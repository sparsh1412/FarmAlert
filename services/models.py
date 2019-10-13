from django.db import models

# Create your models here.
class SoilInput(models.Model):
	Ph = models.FloatField()
	Nitrogen = models.FloatField()
	Phosphorus = models.FloatField()
	Pottasium  = models.FloatField()
	Temprature = models.FloatField()