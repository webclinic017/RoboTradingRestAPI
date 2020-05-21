# Generated by Django 3.0.6 on 2020-05-21 14:48

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('Ticker', '0002_auto_20200521_1401'),
    ]

    operations = [
        migrations.RemoveField(
            model_name='indicator',
            name='id',
        ),
        migrations.AlterField(
            model_name='indicator',
            name='ticker',
            field=models.OneToOneField(on_delete=django.db.models.deletion.CASCADE, primary_key=True, related_name='indicators', serialize=False, to='Ticker.Ticker'),
        ),
    ]