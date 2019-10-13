from django import forms
from django.forms import ModelForm
from services.models import SoilInput

class SoilForm(ModelForm):
	class Meta:
		model = SoilInput
		fields = "__all__"
		
