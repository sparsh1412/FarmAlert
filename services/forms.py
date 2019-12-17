from django import forms
from django.forms import ModelForm
from services.models import SoilInput,YieldInput

class SoilForm(ModelForm):
	class Meta:
		model = SoilInput
		fields = "__all__"
		
class YieldForm(ModelForm):
	class Meta:
		model = YieldInput
		fields = ['Crop','Location']
		
