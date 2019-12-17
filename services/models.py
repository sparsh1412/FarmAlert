from django.db import models

# Create your models here.

CROP_CHOICES= (
    ('Barley', 'Barley'),
    ('Wheat', 'Wheat'),
    ('Maize', 'Maize'),
    ('Sugarcane', 'Sugarcane'),
    ('Rice','Rice'),
    )

LOCATION = (('AMRITSAR','Amritsar'),
             ('BARNALA','Barnala'),
             ('FARIDKOT','Faridkot'),
             ('FAZILKA','Fazilka'),
             ('FIROZEPUR','Firozepur'),
             ('GURDASPUR','Gurdaspur'),
             ('HOSHIARPUR','Hoshiarpur'),
             ('JALANDHAR','Jalandhar'),
             ('KAPURTHALA','Kapurthala'),
             ('LUDHIANA','Ludhiana'),
             ('MANSA','Mansa'),
             ('MOGA','Moga'),
             ('MUKTSAR','Muktsar'),
             ('NAWANSHAHR','Nawanshahr'),
             ('PATHANKOT','Pathankot'),
             ('PATIALA','Patiala'),
             ('RUPNAGAR','Rupnagar'),
             ('SANGRUR','Sangrur'),)

class SoilInput(models.Model):
	Ph = models.FloatField()
	Nitrogen = models.FloatField()
	Phosphorus = models.FloatField()
	Potassium  = models.FloatField()
	Temperature = models.FloatField()

class YieldInput(models.Model):
	Crop = models.CharField(max_length=6, choices=CROP_CHOICES, default='Barley')
	Location = models.CharField(max_length =19,choices=LOCATION,default='PATIALA')