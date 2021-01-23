import requests
import json

#Replace scoring URI and key with their corresponding values. 
scoring_uri = 'http://36c3231c-988a-4252-8ab3-8798c041165e.southcentralus.azurecontainer.io/score'
key = 'RpFLxIGW4uMcLZlKU0iy4JcOa04YVd8c'

# Two sets of data to score, so we get two results back
data = {"data":
        [
          {
            "Column1":0,
            "Age_2.0":1,
            "Age_3.0":0,
            "Age_4.0":0,
            "Fare_2.0":0, 
            "Fare_3.0":0,
            "Fare_4.0":0,
            "Pclass_2":0,
            "Pclass_3":1,
            "Sex_male":1,
            "Embarked_Q":0,
            "Embarked_S":1,
            "Family_type_Large":0,
            "Family_type_Medium":1
          },
          {
            "Column1":1,
            "Age_2.0":0,
            "Age_3.0":0,
            "Age_4.0":1,
            "Fare_2.0":0, 
            "Fare_3.0":1,
            "Fare_4.0":0,
            "Pclass_2":0,
            "Pclass_3":0,
            "Sex_male":0,
            "Embarked_Q":0,
            "Embarked_S":0,
            "Family_type_Large":0,
            "Family_type_Medium":1
          },
      ]
    }

#Convert to JSON string
input_data = json.dumps(data)
with open("data.json", "w") as _f:
   _f.write(input_data)

#Set the content type
headers = {'Content-Type': 'application/json'}

#If authentication is enabled, set the authorization header
headers['Authorization'] = f'Bearer {key}'

#Make the request and display the response
resp = requests.post(scoring_uri, input_data, headers=headers)
print(resp.json())
