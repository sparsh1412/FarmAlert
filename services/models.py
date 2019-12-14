from django.db import models

# Create your models here.
class SoilInput(models.Model):
	Ph = models.FloatField()
	Nitrogen = models.FloatField()
	Phosphorus = models.FloatField()
	Potassium  = models.FloatField()
	Temperature = models.FloatField()

class YieldInput(models.Model):
	Crop = models.CharField(max_length = 50)
	Location = models.CharField(max_length = 50)