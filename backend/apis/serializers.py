from rest_framework import serializers

from apis.models import History

class UserHistorySerializer(serializers.ModelSerializer):
    class Meta:
        model = History
       # fields = ('action', 'text', 'time',)
        fields = '__all__'