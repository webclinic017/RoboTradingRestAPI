# Generated by Django 3.0.6 on 2020-05-19 11:59

from django.db import migrations, models
import django.db.models.deletion


class Migration(migrations.Migration):

    dependencies = [
        ('Ticker', '0002_auto_20200519_1224'),
    ]

    operations = [
        migrations.AlterField(
            model_name='ticker',
            name='stock',
            field=models.ForeignKey(on_delete=django.db.models.deletion.CASCADE, related_name='ticker', to='Ticker.StockMetaData'),
        ),
    ]
