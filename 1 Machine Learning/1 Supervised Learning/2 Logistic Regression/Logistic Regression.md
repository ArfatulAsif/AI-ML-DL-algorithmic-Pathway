{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 1,
   "id": "bd459898-d7e2-4430-b63c-6554740f259e",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "b3325cb2-d1b8-41e0-a5e0-d0ba44d78c61",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = pd.read_csv(\"Social_Network_Ads.csv\")"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 4,
   "id": "a163468f-7459-490a-81c9-306b462c1ef4",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>User ID</th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Purchased</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>15624510</td>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>19000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>15810944</td>\n",
       "      <td>Male</td>\n",
       "      <td>35</td>\n",
       "      <td>20000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>15668575</td>\n",
       "      <td>Female</td>\n",
       "      <td>26</td>\n",
       "      <td>43000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>15603246</td>\n",
       "      <td>Female</td>\n",
       "      <td>27</td>\n",
       "      <td>57000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>15804002</td>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>76000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>395</th>\n",
       "      <td>15691863</td>\n",
       "      <td>Female</td>\n",
       "      <td>46</td>\n",
       "      <td>41000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396</th>\n",
       "      <td>15706071</td>\n",
       "      <td>Male</td>\n",
       "      <td>51</td>\n",
       "      <td>23000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>397</th>\n",
       "      <td>15654296</td>\n",
       "      <td>Female</td>\n",
       "      <td>50</td>\n",
       "      <td>20000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>398</th>\n",
       "      <td>15755018</td>\n",
       "      <td>Male</td>\n",
       "      <td>36</td>\n",
       "      <td>33000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>15594041</td>\n",
       "      <td>Female</td>\n",
       "      <td>49</td>\n",
       "      <td>36000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>400 rows × 5 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "      User ID  Gender  Age  EstimatedSalary  Purchased\n",
       "0    15624510    Male   19            19000          0\n",
       "1    15810944    Male   35            20000          0\n",
       "2    15668575  Female   26            43000          0\n",
       "3    15603246  Female   27            57000          0\n",
       "4    15804002    Male   19            76000          0\n",
       "..        ...     ...  ...              ...        ...\n",
       "395  15691863  Female   46            41000          1\n",
       "396  15706071    Male   51            23000          1\n",
       "397  15654296  Female   50            20000          1\n",
       "398  15755018    Male   36            33000          0\n",
       "399  15594041  Female   49            36000          1\n",
       "\n",
       "[400 rows x 5 columns]"
      ]
     },
     "execution_count": 4,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 5,
   "id": "49007992-3de8-44d7-a48b-f60d7ccc99cd",
   "metadata": {},
   "outputs": [],
   "source": [
    "df = df.drop(columns = 'User ID')   # used to delete a column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 7,
   "id": "a9fdc257-bf88-48bd-b732-c4f8fe5f72be",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Purchased</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>19000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>Male</td>\n",
       "      <td>35</td>\n",
       "      <td>20000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>Female</td>\n",
       "      <td>26</td>\n",
       "      <td>43000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>Female</td>\n",
       "      <td>27</td>\n",
       "      <td>57000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>Male</td>\n",
       "      <td>19</td>\n",
       "      <td>76000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>395</th>\n",
       "      <td>Female</td>\n",
       "      <td>46</td>\n",
       "      <td>41000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396</th>\n",
       "      <td>Male</td>\n",
       "      <td>51</td>\n",
       "      <td>23000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>397</th>\n",
       "      <td>Female</td>\n",
       "      <td>50</td>\n",
       "      <td>20000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>398</th>\n",
       "      <td>Male</td>\n",
       "      <td>36</td>\n",
       "      <td>33000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>Female</td>\n",
       "      <td>49</td>\n",
       "      <td>36000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>400 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Gender  Age  EstimatedSalary  Purchased\n",
       "0      Male   19            19000          0\n",
       "1      Male   35            20000          0\n",
       "2    Female   26            43000          0\n",
       "3    Female   27            57000          0\n",
       "4      Male   19            76000          0\n",
       "..      ...  ...              ...        ...\n",
       "395  Female   46            41000          1\n",
       "396    Male   51            23000          1\n",
       "397  Female   50            20000          1\n",
       "398    Male   36            33000          0\n",
       "399  Female   49            36000          1\n",
       "\n",
       "[400 rows x 4 columns]"
      ]
     },
     "execution_count": 7,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df # Print without 'User ID' column"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 8,
   "id": "e7e577c7-c126-4bbd-a05d-1cf72814476f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "Gender             0\n",
       "Age                0\n",
       "EstimatedSalary    0\n",
       "Purchased          0\n",
       "dtype: int64"
      ]
     },
     "execution_count": 8,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df.isnull().sum()   # For check how many null value available. If want to drop null value command -> df.dropna()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 9,
   "id": "cbedfb93-d7c5-4eee-875c-4c97aeb8310a",
   "metadata": {},
   "outputs": [],
   "source": [
    "# Now we should change all String to number as category ways"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 10,
   "id": "9fd87949-6ea3-4b1f-80fd-0b7c50275943",
   "metadata": {},
   "outputs": [],
   "source": [
    "df['Gender'] = df['Gender'].astype('category')\n",
    "df['Gender'] = df['Gender'].cat.codes"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 11,
   "id": "acfa667c-fb42-4ac9-81b3-96dc190b5d8e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "      <th>Purchased</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>19</td>\n",
       "      <td>19000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>20000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>43000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>27</td>\n",
       "      <td>57000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>19</td>\n",
       "      <td>76000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>395</th>\n",
       "      <td>0</td>\n",
       "      <td>46</td>\n",
       "      <td>41000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396</th>\n",
       "      <td>1</td>\n",
       "      <td>51</td>\n",
       "      <td>23000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>397</th>\n",
       "      <td>0</td>\n",
       "      <td>50</td>\n",
       "      <td>20000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>398</th>\n",
       "      <td>1</td>\n",
       "      <td>36</td>\n",
       "      <td>33000</td>\n",
       "      <td>0</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>0</td>\n",
       "      <td>49</td>\n",
       "      <td>36000</td>\n",
       "      <td>1</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>400 rows × 4 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Gender  Age  EstimatedSalary  Purchased\n",
       "0         1   19            19000          0\n",
       "1         1   35            20000          0\n",
       "2         0   26            43000          0\n",
       "3         0   27            57000          0\n",
       "4         1   19            76000          0\n",
       "..      ...  ...              ...        ...\n",
       "395       0   46            41000          1\n",
       "396       1   51            23000          1\n",
       "397       0   50            20000          1\n",
       "398       1   36            33000          0\n",
       "399       0   49            36000          1\n",
       "\n",
       "[400 rows x 4 columns]"
      ]
     },
     "execution_count": 11,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "df"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 12,
   "id": "27ada666-7b67-4402-b1a9-be2f44ccc49f",
   "metadata": {},
   "outputs": [],
   "source": [
    "X = df.drop(columns = 'Purchased')  # Pick the independent variables"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 13,
   "id": "64c5e5f2-87ea-45d3-a5d7-32b2fdf56f9e",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>0</th>\n",
       "      <td>1</td>\n",
       "      <td>19</td>\n",
       "      <td>19000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>1</th>\n",
       "      <td>1</td>\n",
       "      <td>35</td>\n",
       "      <td>20000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>2</th>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>43000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>3</th>\n",
       "      <td>0</td>\n",
       "      <td>27</td>\n",
       "      <td>57000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>4</th>\n",
       "      <td>1</td>\n",
       "      <td>19</td>\n",
       "      <td>76000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>395</th>\n",
       "      <td>0</td>\n",
       "      <td>46</td>\n",
       "      <td>41000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>396</th>\n",
       "      <td>1</td>\n",
       "      <td>51</td>\n",
       "      <td>23000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>397</th>\n",
       "      <td>0</td>\n",
       "      <td>50</td>\n",
       "      <td>20000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>398</th>\n",
       "      <td>1</td>\n",
       "      <td>36</td>\n",
       "      <td>33000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>399</th>\n",
       "      <td>0</td>\n",
       "      <td>49</td>\n",
       "      <td>36000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>400 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Gender  Age  EstimatedSalary\n",
       "0         1   19            19000\n",
       "1         1   35            20000\n",
       "2         0   26            43000\n",
       "3         0   27            57000\n",
       "4         1   19            76000\n",
       "..      ...  ...              ...\n",
       "395       0   46            41000\n",
       "396       1   51            23000\n",
       "397       0   50            20000\n",
       "398       1   36            33000\n",
       "399       0   49            36000\n",
       "\n",
       "[400 rows x 3 columns]"
      ]
     },
     "execution_count": 13,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 14,
   "id": "b6fca66a-89d3-4ec7-925d-5aa7bfcd968c",
   "metadata": {},
   "outputs": [],
   "source": [
    "Y = df['Purchased']  # Pick the Predicted Variable"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 15,
   "id": "e72c4f84-5688-4640-a9bb-ca665783d6b2",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0      0\n",
       "1      0\n",
       "2      0\n",
       "3      0\n",
       "4      0\n",
       "      ..\n",
       "395    1\n",
       "396    1\n",
       "397    1\n",
       "398    0\n",
       "399    1\n",
       "Name: Purchased, Length: 400, dtype: int64"
      ]
     },
     "execution_count": 15,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 16,
   "id": "a02b3b0d-3263-4bb9-a652-3451ea1e4fd7",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.model_selection import train_test_split "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 17,
   "id": "2f9faabe-8fa8-42ab-90e4-e1956a1b22f1",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size = 0.3, random_state = 21) # test_size = 0.3 means 30% will be used for test and 70% for training. random_state can be any integer, by using random_state we can find same train-test_split "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 18,
   "id": "2cbc9e7d-0676-420d-a143-623e7c016ddf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>113</th>\n",
       "      <td>1</td>\n",
       "      <td>37</td>\n",
       "      <td>55000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>26</th>\n",
       "      <td>1</td>\n",
       "      <td>49</td>\n",
       "      <td>28000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>178</th>\n",
       "      <td>1</td>\n",
       "      <td>24</td>\n",
       "      <td>23000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>95</th>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>44000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>29</th>\n",
       "      <td>1</td>\n",
       "      <td>31</td>\n",
       "      <td>18000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>368</th>\n",
       "      <td>1</td>\n",
       "      <td>38</td>\n",
       "      <td>71000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>48</th>\n",
       "      <td>1</td>\n",
       "      <td>30</td>\n",
       "      <td>135000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>260</th>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>77000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>312</th>\n",
       "      <td>0</td>\n",
       "      <td>38</td>\n",
       "      <td>50000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>207</th>\n",
       "      <td>0</td>\n",
       "      <td>52</td>\n",
       "      <td>114000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>280 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Gender  Age  EstimatedSalary\n",
       "113       1   37            55000\n",
       "26        1   49            28000\n",
       "178       1   24            23000\n",
       "95        0   35            44000\n",
       "29        1   31            18000\n",
       "..      ...  ...              ...\n",
       "368       1   38            71000\n",
       "48        1   30           135000\n",
       "260       0   35            77000\n",
       "312       0   38            50000\n",
       "207       0   52           114000\n",
       "\n",
       "[280 rows x 3 columns]"
      ]
     },
     "execution_count": 18,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 19,
   "id": "d36f399b-c744-4b74-8337-8eff7026e796",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/html": [
       "<div>\n",
       "<style scoped>\n",
       "    .dataframe tbody tr th:only-of-type {\n",
       "        vertical-align: middle;\n",
       "    }\n",
       "\n",
       "    .dataframe tbody tr th {\n",
       "        vertical-align: top;\n",
       "    }\n",
       "\n",
       "    .dataframe thead th {\n",
       "        text-align: right;\n",
       "    }\n",
       "</style>\n",
       "<table border=\"1\" class=\"dataframe\">\n",
       "  <thead>\n",
       "    <tr style=\"text-align: right;\">\n",
       "      <th></th>\n",
       "      <th>Gender</th>\n",
       "      <th>Age</th>\n",
       "      <th>EstimatedSalary</th>\n",
       "    </tr>\n",
       "  </thead>\n",
       "  <tbody>\n",
       "    <tr>\n",
       "      <th>106</th>\n",
       "      <td>0</td>\n",
       "      <td>26</td>\n",
       "      <td>35000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>9</th>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>65000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>61</th>\n",
       "      <td>1</td>\n",
       "      <td>25</td>\n",
       "      <td>87000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>224</th>\n",
       "      <td>0</td>\n",
       "      <td>35</td>\n",
       "      <td>60000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>37</th>\n",
       "      <td>1</td>\n",
       "      <td>30</td>\n",
       "      <td>49000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>...</th>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "      <td>...</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>23</th>\n",
       "      <td>0</td>\n",
       "      <td>45</td>\n",
       "      <td>22000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>157</th>\n",
       "      <td>1</td>\n",
       "      <td>29</td>\n",
       "      <td>75000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>349</th>\n",
       "      <td>1</td>\n",
       "      <td>38</td>\n",
       "      <td>61000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>255</th>\n",
       "      <td>0</td>\n",
       "      <td>52</td>\n",
       "      <td>90000</td>\n",
       "    </tr>\n",
       "    <tr>\n",
       "      <th>180</th>\n",
       "      <td>1</td>\n",
       "      <td>26</td>\n",
       "      <td>16000</td>\n",
       "    </tr>\n",
       "  </tbody>\n",
       "</table>\n",
       "<p>120 rows × 3 columns</p>\n",
       "</div>"
      ],
      "text/plain": [
       "     Gender  Age  EstimatedSalary\n",
       "106       0   26            35000\n",
       "9         0   35            65000\n",
       "61        1   25            87000\n",
       "224       0   35            60000\n",
       "37        1   30            49000\n",
       "..      ...  ...              ...\n",
       "23        0   45            22000\n",
       "157       1   29            75000\n",
       "349       1   38            61000\n",
       "255       0   52            90000\n",
       "180       1   26            16000\n",
       "\n",
       "[120 rows x 3 columns]"
      ]
     },
     "execution_count": 19,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 20,
   "id": "c9f87cc0-7cb5-4343-b4b5-a3bf44563335",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "113    0\n",
       "26     1\n",
       "178    0\n",
       "95     0\n",
       "29     0\n",
       "      ..\n",
       "368    0\n",
       "48     1\n",
       "260    0\n",
       "312    0\n",
       "207    0\n",
       "Name: Purchased, Length: 280, dtype: int64"
      ]
     },
     "execution_count": 20,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_train"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 21,
   "id": "b33548a5-b53e-4954-8261-54e8a7999fc0",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "106    0\n",
       "9      0\n",
       "61     0\n",
       "224    0\n",
       "37     0\n",
       "      ..\n",
       "23     1\n",
       "157    0\n",
       "349    0\n",
       "255    1\n",
       "180    0\n",
       "Name: Purchased, Length: 120, dtype: int64"
      ]
     },
     "execution_count": 21,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "Y_test"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 22,
   "id": "144428c5-6fbd-48a2-9a90-5e85383c5db6",
   "metadata": {},
   "outputs": [],
   "source": [
    "# For all numbers into an fixed range we can do some Scalling for pre-processing"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 23,
   "id": "c4bb2d5e-c143-4a5c-8a71-a11c287538e8",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.preprocessing import StandardScaler "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 24,
   "id": "caf86fbd-095a-40a8-a6f3-95e47524d2ed",
   "metadata": {},
   "outputs": [],
   "source": [
    "scaler = StandardScaler()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 28,
   "id": "44a1ae8b-2974-4fb3-b410-5afb39129f7c",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_train_scaled = scaler.fit_transform(X_train) # Use fit_transform() to know the train sets 'standard deviation' and 'mean'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 27,
   "id": "7cd11193-4bd7-4d57-b35d-d3c632a4bd47",
   "metadata": {},
   "outputs": [],
   "source": [
    "X_test_scaled = scaler.transform(X_test) # Use only transform() so that model should not know anything about 'test set'"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 29,
   "id": "14774b33-048b-4162-b82b-143d840e279c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[ 0.97182532, -0.11728762, -0.48154649],\n",
       "       [ 0.97182532,  1.06615502, -1.26799962],\n",
       "       [ 0.97182532, -1.39935048, -1.41363908],\n",
       "       [-1.02899151, -0.31452806, -0.80195332],\n",
       "       [ 0.97182532, -0.70900894, -1.55927855],\n",
       "       [ 0.97182532, -1.0048696 ,  0.21752295],\n",
       "       [-1.02899151,  0.4744337 ,  1.79042919],\n",
       "       [ 0.97182532, -0.21590784, -0.33590703],\n",
       "       [-1.02899151, -1.0048696 ,  0.36316241],\n",
       "       [-1.02899151, -0.31452806, -1.41363908],\n",
       "       [-1.02899151, -0.61038872,  0.4214182 ],\n",
       "       [ 0.97182532, -1.9910718 , -0.56893017],\n",
       "       [ 0.97182532, -0.31452806, -1.44276698],\n",
       "       [ 0.97182532, -1.0048696 ,  1.49915025],\n",
       "       [ 0.97182532, -1.39935048, -0.48154649],\n",
       "       [ 0.97182532, -1.30073026,  0.21752295],\n",
       "       [ 0.97182532, -1.20211004,  0.24665084],\n",
       "       [-1.02899151,  0.86891458,  1.03310396],\n",
       "       [ 0.97182532, -0.70900894,  0.07188348],\n",
       "       [ 0.97182532,  2.15097745,  0.88746449],\n",
       "       [-1.02899151, -0.61038872,  1.32438289],\n",
       "       [ 0.97182532, -0.0186674 , -0.0155002 ],\n",
       "       [ 0.97182532,  1.75649657, -0.33590703],\n",
       "       [ 0.97182532,  0.17857304,  1.03310396],\n",
       "       [ 0.97182532, -1.10348982, -1.50102276],\n",
       "       [-1.02899151,  1.95373701,  0.68356924],\n",
       "       [ 0.97182532,  0.37581348, -0.21939545],\n",
       "       [-1.02899151,  0.27719326, -0.59805807],\n",
       "       [-1.02899151,  1.85511679, -1.12236015],\n",
       "       [-1.02899151,  0.77029436,  0.07188348],\n",
       "       [-1.02899151, -0.21590784, -0.51067439],\n",
       "       [-1.02899151,  2.05235723, -0.86020911],\n",
       "       [ 0.97182532, -0.31452806, -0.97672068],\n",
       "       [ 0.97182532,  2.05235723,  2.08170812],\n",
       "       [-1.02899151, -0.61038872,  1.84868498],\n",
       "       [-1.02899151, -0.31452806, -0.42329071],\n",
       "       [-1.02899151, -0.0186674 ,  1.17874343],\n",
       "       [ 0.97182532, -1.39935048,  0.36316241],\n",
       "       [-1.02899151,  2.15097745,  1.06223185],\n",
       "       [ 0.97182532,  1.06615502,  0.07188348],\n",
       "       [-1.02899151,  0.37581348,  1.06223185],\n",
       "       [-1.02899151,  0.17857304,  2.05258023],\n",
       "       [ 0.97182532, -1.20211004,  0.27577873],\n",
       "       [-1.02899151, -1.20211004,  0.36316241],\n",
       "       [ 0.97182532,  0.07995282, -0.86020911],\n",
       "       [ 0.97182532, -0.31452806,  0.21752295],\n",
       "       [-1.02899151, -1.4979707 , -0.68544175],\n",
       "       [-1.02899151,  1.16477524, -1.03497647],\n",
       "       [ 0.97182532,  0.86891458, -1.50102276],\n",
       "       [-1.02899151, -1.4979707 , -0.16113967],\n",
       "       [ 0.97182532,  1.85511679,  0.07188348],\n",
       "       [ 0.97182532,  0.9675348 , -1.12236015],\n",
       "       [ 0.97182532,  1.95373701,  2.11083602],\n",
       "       [-1.02899151, -0.5117685 ,  1.20787132],\n",
       "       [-1.02899151, -1.10348982, -1.18061594],\n",
       "       [ 0.97182532, -0.21590784, -0.56893017],\n",
       "       [ 0.97182532, -0.90624938,  0.24665084],\n",
       "       [ 0.97182532,  0.86891458, -0.83108121],\n",
       "       [ 0.97182532, -1.69521114,  0.01362769],\n",
       "       [ 0.97182532, -0.61038872, -1.55927855],\n",
       "       [-1.02899151,  0.86891458, -0.65631385],\n",
       "       [ 0.97182532, -0.11728762, -0.53980228],\n",
       "       [ 0.97182532,  0.9675348 ,  0.53792977],\n",
       "       [ 0.97182532,  0.67167414, -1.3262554 ],\n",
       "       [-1.02899151, -1.20211004, -1.64666223],\n",
       "       [ 0.97182532, -0.31452806,  0.10101137],\n",
       "       [ 0.97182532,  0.86891458, -1.41363908],\n",
       "       [-1.02899151,  0.9675348 ,  1.38263868],\n",
       "       [-1.02899151, -1.0048696 , -0.36503492],\n",
       "       [-1.02899151, -0.41314828, -0.83108121],\n",
       "       [-1.02899151, -0.11728762, -0.0155002 ],\n",
       "       [-1.02899151, -0.61038872,  2.28560338],\n",
       "       [-1.02899151, -1.0048696 ,  0.45054609],\n",
       "       [ 0.97182532, -0.31452806, -0.53980228],\n",
       "       [ 0.97182532,  1.06615502,  0.47967399],\n",
       "       [ 0.97182532,  0.07995282,  1.81955708],\n",
       "       [-1.02899151, -1.0048696 ,  0.39229031],\n",
       "       [-1.02899151, -0.0186674 ,  0.24665084],\n",
       "       [ 0.97182532, -0.31452806,  0.47967399],\n",
       "       [ 0.97182532, -1.30073026, -1.44276698],\n",
       "       [-1.02899151,  0.27719326,  0.01362769],\n",
       "       [ 0.97182532, -0.41314828,  1.17874343],\n",
       "       [-1.02899151,  1.55925613, -1.3262554 ],\n",
       "       [-1.02899151, -1.20211004, -0.83108121],\n",
       "       [-1.02899151,  0.9675348 ,  1.81955708],\n",
       "       [ 0.97182532, -0.31452806,  1.06223185],\n",
       "       [-1.02899151, -1.20211004, -1.58840644],\n",
       "       [ 0.97182532, -0.61038872, -1.55927855],\n",
       "       [-1.02899151, -0.70900894,  1.35351079],\n",
       "       [-1.02899151, -0.31452806,  0.01362769],\n",
       "       [ 0.97182532, -0.11728762, -0.56893017],\n",
       "       [ 0.97182532, -0.11728762,  2.11083602],\n",
       "       [-1.02899151, -1.79383136, -1.03497647],\n",
       "       [ 0.97182532,  1.65787635, -0.94759279],\n",
       "       [ 0.97182532,  0.86891458,  0.97484817],\n",
       "       [-1.02899151,  1.16477524, -1.50102276],\n",
       "       [-1.02899151,  0.67167414, -0.77282543],\n",
       "       [-1.02899151,  0.27719326, -1.20974383],\n",
       "       [ 0.97182532, -1.89245158, -1.3553833 ],\n",
       "       [-1.02899151, -0.11728762, -0.42329071],\n",
       "       [ 0.97182532,  0.9675348 , -0.889337  ],\n",
       "       [-1.02899151,  1.36201569, -1.47189487],\n",
       "       [ 0.97182532, -1.10348982,  0.47967399],\n",
       "       [ 0.97182532,  0.67167414, -1.15148804],\n",
       "       [ 0.97182532,  0.9675348 , -1.12236015],\n",
       "       [-1.02899151, -1.10348982, -0.39416281],\n",
       "       [-1.02899151, -0.70900894, -0.10288388],\n",
       "       [ 0.97182532,  0.67167414,  0.21752295],\n",
       "       [ 0.97182532, -1.79383136,  0.4214182 ],\n",
       "       [-1.02899151,  1.46063591, -1.09323225],\n",
       "       [-1.02899151, -1.20211004,  1.35351079],\n",
       "       [ 0.97182532,  0.17857304, -0.71456964],\n",
       "       [ 0.97182532, -0.5117685 , -1.18061594],\n",
       "       [-1.02899151,  1.95373701, -1.41363908],\n",
       "       [ 0.97182532,  0.37581348,  0.94572028],\n",
       "       [ 0.97182532, -1.9910718 ,  0.30490663],\n",
       "       [ 0.97182532,  0.37581348,  0.24665084],\n",
       "       [-1.02899151, -1.10348982,  0.71269713],\n",
       "       [ 0.97182532,  0.17857304,  0.18839505],\n",
       "       [-1.02899151,  0.4744337 ,  1.17874343],\n",
       "       [-1.02899151,  1.95373701, -0.71456964],\n",
       "       [-1.02899151, -1.10348982,  1.90694076],\n",
       "       [ 0.97182532, -1.59659092,  0.27577873],\n",
       "       [-1.02899151,  0.9675348 ,  0.71269713],\n",
       "       [-1.02899151, -0.21590784,  1.58653393],\n",
       "       [ 0.97182532, -0.31452806, -1.29712751],\n",
       "       [ 0.97182532,  0.77029436,  0.47967399],\n",
       "       [ 0.97182532, -1.89245158,  0.39229031],\n",
       "       [ 0.97182532, -1.0048696 , -0.36503492],\n",
       "       [-1.02899151,  0.27719326,  0.01362769],\n",
       "       [-1.02899151, -0.31452806, -0.0155002 ],\n",
       "       [ 0.97182532, -0.80762916, -1.64666223],\n",
       "       [ 0.97182532,  0.27719326, -0.77282543],\n",
       "       [ 0.97182532,  0.9675348 ,  0.07188348],\n",
       "       [ 0.97182532,  0.07995282,  1.00397607],\n",
       "       [ 0.97182532, -1.20211004,  0.4214182 ],\n",
       "       [ 0.97182532, -0.11728762, -0.04462809],\n",
       "       [-1.02899151,  0.67167414,  1.7321734 ],\n",
       "       [-1.02899151, -0.0186674 , -0.19026756],\n",
       "       [ 0.97182532,  0.17857304, -0.0155002 ],\n",
       "       [-1.02899151,  0.27719326, -0.24852335],\n",
       "       [-1.02899151, -0.0186674 ,  1.20787132],\n",
       "       [-1.02899151, -0.5117685 , -1.26799962],\n",
       "       [ 0.97182532, -0.0186674 , -0.59805807],\n",
       "       [ 0.97182532,  0.77029436,  1.32438289],\n",
       "       [ 0.97182532, -1.20211004, -1.20974383],\n",
       "       [ 0.97182532,  2.05235723,  1.70304551],\n",
       "       [ 0.97182532, -1.89245158, -1.53015066],\n",
       "       [-1.02899151, -0.5117685 , -0.59805807],\n",
       "       [ 0.97182532, -1.10348982,  0.53792977],\n",
       "       [ 0.97182532, -0.21590784, -1.12236015],\n",
       "       [ 0.97182532,  2.15097745, -1.09323225],\n",
       "       [-1.02899151, -1.59659092, -0.24852335],\n",
       "       [-1.02899151, -1.20211004,  0.24665084],\n",
       "       [-1.02899151,  0.07995282, -0.36503492],\n",
       "       [-1.02899151, -0.11728762,  0.24665084],\n",
       "       [ 0.97182532, -1.0048696 , -1.15148804],\n",
       "       [-1.02899151,  0.37581348,  0.21752295],\n",
       "       [-1.02899151,  0.9675348 ,  1.7321734 ],\n",
       "       [ 0.97182532,  0.17857304, -0.42329071],\n",
       "       [ 0.97182532, -1.89245158, -0.04462809],\n",
       "       [-1.02899151, -1.20211004,  0.01362769],\n",
       "       [-1.02899151, -0.90624938,  0.33403452],\n",
       "       [-1.02899151, -1.10348982, -0.42329071],\n",
       "       [-1.02899151,  0.37581348,  0.10101137],\n",
       "       [ 0.97182532,  0.07995282, -0.86020911],\n",
       "       [-1.02899151,  0.07995282, -0.30677913],\n",
       "       [-1.02899151,  2.15097745, -0.74369753],\n",
       "       [ 0.97182532, -0.5117685 , -0.83108121],\n",
       "       [ 0.97182532,  0.07995282, -0.0155002 ],\n",
       "       [ 0.97182532,  0.37581348, -0.51067439],\n",
       "       [-1.02899151,  0.86891458, -0.71456964],\n",
       "       [ 0.97182532, -0.11728762,  0.15926716],\n",
       "       [ 0.97182532, -0.11728762,  0.07188348],\n",
       "       [ 0.97182532,  1.85511679, -0.33590703],\n",
       "       [-1.02899151, -0.70900894,  0.50880188],\n",
       "       [-1.02899151,  1.26339546,  1.81955708],\n",
       "       [ 0.97182532,  0.07995282,  1.47002236],\n",
       "       [ 0.97182532, -1.39935048, -0.39416281],\n",
       "       [ 0.97182532, -0.21590784,  1.55740604],\n",
       "       [-1.02899151, -1.4979707 ,  0.30490663],\n",
       "       [ 0.97182532, -0.31452806,  0.56705767],\n",
       "       [ 0.97182532, -1.0048696 ,  0.50880188],\n",
       "       [-1.02899151,  0.9675348 ,  1.93606866],\n",
       "       [-1.02899151, -1.69521114, -1.61753434],\n",
       "       [ 0.97182532,  1.06615502, -1.26799962],\n",
       "       [ 0.97182532, -1.30073026,  0.53792977],\n",
       "       [-1.02899151,  0.86891458, -1.20974383],\n",
       "       [-1.02899151,  0.37581348, -0.04462809],\n",
       "       [-1.02899151,  0.86891458,  1.20787132],\n",
       "       [ 0.97182532, -0.31452806, -1.50102276],\n",
       "       [ 0.97182532,  1.75649657,  1.79042919],\n",
       "       [-1.02899151, -0.41314828, -1.3553833 ],\n",
       "       [-1.02899151,  0.77029436, -0.889337  ],\n",
       "       [-1.02899151, -0.11728762, -1.12236015],\n",
       "       [ 0.97182532, -0.80762916,  0.45054609],\n",
       "       [-1.02899151,  1.16477524, -0.80195332],\n",
       "       [-1.02899151,  1.26339546,  2.1690918 ],\n",
       "       [-1.02899151, -0.21590784, -0.24852335],\n",
       "       [ 0.97182532,  0.77029436, -1.26799962],\n",
       "       [ 0.97182532, -0.41314828,  1.26612711],\n",
       "       [ 0.97182532, -1.10348982, -0.39416281],\n",
       "       [ 0.97182532, -0.11728762,  0.21752295],\n",
       "       [-1.02899151,  1.65787635,  1.70304551],\n",
       "       [-1.02899151,  0.27719326,  0.24665084],\n",
       "       [-1.02899151, -0.21590784, -0.62718596],\n",
       "       [-1.02899151,  2.05235723,  0.33403452],\n",
       "       [ 0.97182532, -1.10348982,  0.50880188],\n",
       "       [ 0.97182532,  0.37581348, -0.19026756],\n",
       "       [ 0.97182532, -0.31452806, -0.30677913],\n",
       "       [-1.02899151,  2.05235723,  0.47967399],\n",
       "       [-1.02899151, -0.5117685 ,  2.25647548],\n",
       "       [ 0.97182532, -1.59659092, -1.55927855],\n",
       "       [ 0.97182532, -1.30073026, -1.12236015],\n",
       "       [-1.02899151, -1.0048696 , -0.80195332],\n",
       "       [-1.02899151, -1.39935048, -1.29712751],\n",
       "       [ 0.97182532, -0.11728762,  0.01362769],\n",
       "       [ 0.97182532,  0.27719326, -0.56893017],\n",
       "       [-1.02899151, -0.70900894, -0.0155002 ],\n",
       "       [-1.02899151,  1.06615502, -1.03497647],\n",
       "       [ 0.97182532, -0.61038872,  1.41176657],\n",
       "       [ 0.97182532, -0.0186674 , -0.30677913],\n",
       "       [ 0.97182532, -0.61038872,  0.82920871],\n",
       "       [-1.02899151,  0.57305392,  1.96519655],\n",
       "       [-1.02899151,  0.77029436,  0.71269713],\n",
       "       [-1.02899151,  0.57305392, -0.94759279],\n",
       "       [-1.02899151, -0.31452806,  2.1982197 ],\n",
       "       [ 0.97182532,  0.07995282,  0.15926716],\n",
       "       [-1.02899151, -0.11728762, -0.27765124],\n",
       "       [ 0.97182532,  0.37581348, -0.19026756],\n",
       "       [ 0.97182532,  0.17857304, -0.36503492],\n",
       "       [ 0.97182532,  0.77029436,  0.21752295],\n",
       "       [-1.02899151,  0.27719326, -0.33590703],\n",
       "       [ 0.97182532,  2.15097745,  0.33403452],\n",
       "       [ 0.97182532, -0.80762916,  1.03310396],\n",
       "       [ 0.97182532, -0.90624938, -0.83108121],\n",
       "       [ 0.97182532,  1.46063591,  0.01362769],\n",
       "       [-1.02899151,  2.05235723,  0.13013927],\n",
       "       [-1.02899151, -0.0186674 , -0.48154649],\n",
       "       [ 0.97182532,  0.37581348,  2.25647548],\n",
       "       [-1.02899151,  0.86891458,  2.11083602],\n",
       "       [ 0.97182532,  0.27719326, -0.56893017],\n",
       "       [ 0.97182532,  1.06615502,  0.50880188],\n",
       "       [ 0.97182532, -0.80762916,  0.24665084],\n",
       "       [ 0.97182532, -0.21590784,  2.11083602],\n",
       "       [-1.02899151, -0.0186674 , -0.62718596],\n",
       "       [ 0.97182532,  1.26339546, -1.41363908],\n",
       "       [ 0.97182532,  0.17857304, -0.42329071],\n",
       "       [-1.02899151, -1.39935048, -1.15148804],\n",
       "       [-1.02899151,  1.85511679, -1.3262554 ],\n",
       "       [-1.02899151,  0.37581348,  0.24665084],\n",
       "       [-1.02899151, -1.69521114, -0.10288388],\n",
       "       [-1.02899151, -0.90624938, -0.71456964],\n",
       "       [-1.02899151, -0.5117685 , -0.33590703],\n",
       "       [-1.02899151,  1.95373701,  0.8583366 ],\n",
       "       [-1.02899151,  0.77029436, -1.15148804],\n",
       "       [ 0.97182532, -0.90624938, -0.30677913],\n",
       "       [-1.02899151,  0.07995282,  0.10101137],\n",
       "       [-1.02899151, -0.80762916, -0.27765124],\n",
       "       [ 0.97182532,  1.55925613, -0.04462809],\n",
       "       [-1.02899151,  0.77029436,  0.30490663],\n",
       "       [-1.02899151,  0.17857304, -0.33590703],\n",
       "       [-1.02899151,  0.17857304,  0.10101137],\n",
       "       [-1.02899151, -0.31452806, -0.62718596],\n",
       "       [-1.02899151,  1.06615502, -0.94759279],\n",
       "       [ 0.97182532,  1.06615502,  0.4214182 ],\n",
       "       [-1.02899151, -0.11728762,  2.1690918 ],\n",
       "       [ 0.97182532, -0.31452806, -0.48154649],\n",
       "       [-1.02899151,  0.07995282, -0.0155002 ],\n",
       "       [ 0.97182532, -0.21590784,  1.35351079],\n",
       "       [-1.02899151,  0.07995282,  1.81955708],\n",
       "       [ 0.97182532, -0.0186674 , -0.36503492],\n",
       "       [ 0.97182532, -0.31452806,  0.04275559],\n",
       "       [ 0.97182532, -0.31452806,  0.01362769],\n",
       "       [-1.02899151, -0.21590784,  0.10101137],\n",
       "       [ 0.97182532, -0.0186674 , -0.0155002 ],\n",
       "       [ 0.97182532, -0.80762916,  1.84868498],\n",
       "       [-1.02899151, -0.31452806,  0.15926716],\n",
       "       [-1.02899151, -0.0186674 , -0.62718596],\n",
       "       [-1.02899151,  1.36201569,  1.23699921]])"
      ]
     },
     "execution_count": 29,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_train_scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 30,
   "id": "bc17cd03-9c5e-4c03-9fec-f3f21e0cb18c",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([[-1.02899151, -1.20211004, -1.06410436],\n",
       "       [-1.02899151, -0.31452806, -0.19026756],\n",
       "       [ 0.97182532, -1.30073026,  0.45054609],\n",
       "       [-1.02899151, -0.31452806, -0.33590703],\n",
       "       [ 0.97182532, -0.80762916, -0.65631385],\n",
       "       [-1.02899151,  1.16477524,  0.47967399],\n",
       "       [-1.02899151,  1.06615502,  2.02345234],\n",
       "       [-1.02899151, -0.90624938,  0.33403452],\n",
       "       [-1.02899151,  1.55925613,  1.06223185],\n",
       "       [-1.02899151,  1.46063591,  2.08170812],\n",
       "       [-1.02899151,  0.07995282,  0.21752295],\n",
       "       [-1.02899151, -1.0048696 , -1.00584857],\n",
       "       [-1.02899151,  0.77029436, -1.44276698],\n",
       "       [ 0.97182532,  0.07995282,  0.71269713],\n",
       "       [ 0.97182532,  0.27719326,  0.01362769],\n",
       "       [-1.02899151, -1.79383136,  0.30490663],\n",
       "       [-1.02899151,  0.07995282,  0.04275559],\n",
       "       [ 0.97182532,  0.86891458, -1.3553833 ],\n",
       "       [ 0.97182532,  0.17857304,  0.10101137],\n",
       "       [-1.02899151, -0.5117685 , -0.07375599],\n",
       "       [-1.02899151, -1.4979707 , -1.26799962],\n",
       "       [ 0.97182532, -1.79383136,  0.07188348],\n",
       "       [ 0.97182532,  0.27719326, -0.36503492],\n",
       "       [-1.02899151, -1.9910718 , -0.80195332],\n",
       "       [ 0.97182532, -0.31452806,  0.10101137],\n",
       "       [ 0.97182532, -0.31452806, -0.62718596],\n",
       "       [ 0.97182532,  1.36201569,  2.28560338],\n",
       "       [ 0.97182532,  2.15097745, -0.86020911],\n",
       "       [-1.02899151, -1.10348982, -0.51067439],\n",
       "       [-1.02899151, -0.41314828,  0.01362769],\n",
       "       [-1.02899151, -1.9910718 , -0.10288388],\n",
       "       [-1.02899151, -1.10348982, -1.58840644],\n",
       "       [ 0.97182532, -1.79383136, -0.65631385],\n",
       "       [ 0.97182532,  0.67167414, -1.44276698],\n",
       "       [ 0.97182532, -1.4979707 , -0.24852335],\n",
       "       [-1.02899151, -0.5117685 , -0.889337  ],\n",
       "       [ 0.97182532,  0.17857304, -0.42329071],\n",
       "       [-1.02899151, -0.31452806, -1.3553833 ],\n",
       "       [-1.02899151, -0.11728762,  0.24665084],\n",
       "       [ 0.97182532,  0.77029436, -0.36503492],\n",
       "       [-1.02899151,  1.75649657,  0.94572028],\n",
       "       [-1.02899151, -1.20211004, -0.56893017],\n",
       "       [ 0.97182532, -1.30073026,  0.24665084],\n",
       "       [-1.02899151, -1.39935048, -0.48154649],\n",
       "       [ 0.97182532, -0.90624938, -0.83108121],\n",
       "       [-1.02899151, -1.79383136, -1.41363908],\n",
       "       [ 0.97182532, -1.39935048, -1.53015066],\n",
       "       [-1.02899151,  1.65787635,  1.55740604],\n",
       "       [ 0.97182532,  0.17857304, -0.19026756],\n",
       "       [ 0.97182532,  0.27719326,  0.45054609],\n",
       "       [ 0.97182532,  0.37581348,  0.04275559],\n",
       "       [ 0.97182532, -0.31452806, -0.39416281],\n",
       "       [ 0.97182532, -0.70900894, -0.39416281],\n",
       "       [-1.02899151, -0.70900894, -1.09323225],\n",
       "       [-1.02899151,  0.86891458, -0.59805807],\n",
       "       [ 0.97182532,  0.86891458, -1.09323225],\n",
       "       [-1.02899151,  0.9675348 , -1.20974383],\n",
       "       [ 0.97182532, -0.21590784,  0.80008081],\n",
       "       [-1.02899151, -1.0048696 , -0.48154649],\n",
       "       [ 0.97182532, -0.90624938,  2.22734759],\n",
       "       [ 0.97182532, -0.31452806, -0.94759279],\n",
       "       [ 0.97182532,  0.4744337 ,  1.67391761],\n",
       "       [-1.02899151,  0.9675348 , -1.06410436],\n",
       "       [-1.02899151, -0.61038872,  1.32438289],\n",
       "       [ 0.97182532,  0.27719326,  0.21752295],\n",
       "       [-1.02899151,  1.85511679,  1.47002236],\n",
       "       [-1.02899151, -0.31452806,  0.74182503],\n",
       "       [ 0.97182532, -1.69521114,  0.47967399],\n",
       "       [ 0.97182532, -0.70900894, -0.16113967],\n",
       "       [-1.02899151,  1.95373701, -0.97672068],\n",
       "       [ 0.97182532, -0.31452806, -0.36503492],\n",
       "       [ 0.97182532, -1.20211004, -1.64666223],\n",
       "       [-1.02899151,  0.17857304,  0.01362769],\n",
       "       [ 0.97182532, -1.20211004, -1.15148804],\n",
       "       [ 0.97182532,  0.9675348 ,  2.02345234],\n",
       "       [ 0.97182532, -1.4979707 , -1.50102276],\n",
       "       [ 0.97182532,  0.37581348, -0.51067439],\n",
       "       [ 0.97182532, -0.11728762,  0.10101137],\n",
       "       [-1.02899151, -1.79383136,  0.30490663],\n",
       "       [-1.02899151,  0.86891458, -0.62718596],\n",
       "       [-1.02899151, -1.89245158, -1.3262554 ],\n",
       "       [-1.02899151,  1.36201569, -0.97672068],\n",
       "       [ 0.97182532,  2.15097745, -0.86020911],\n",
       "       [ 0.97182532, -0.80762916, -1.58840644],\n",
       "       [-1.02899151, -0.11728762,  0.62531345],\n",
       "       [-1.02899151, -1.89245158, -1.47189487],\n",
       "       [ 0.97182532, -0.41314828, -0.83108121],\n",
       "       [-1.02899151, -1.10348982,  0.36316241],\n",
       "       [-1.02899151,  0.27719326, -0.0155002 ],\n",
       "       [-1.02899151, -1.59659092, -1.29712751],\n",
       "       [-1.02899151, -1.59659092, -0.48154649],\n",
       "       [-1.02899151,  1.46063591,  0.94572028],\n",
       "       [-1.02899151,  0.37581348,  0.53792977],\n",
       "       [ 0.97182532,  0.77029436, -1.41363908],\n",
       "       [-1.02899151,  1.55925613,  0.94572028],\n",
       "       [ 0.97182532, -1.89245158,  0.13013927],\n",
       "       [-1.02899151, -1.9910718 ,  0.4214182 ],\n",
       "       [-1.02899151, -0.90624938, -1.26799962],\n",
       "       [-1.02899151, -0.11728762,  0.18839505],\n",
       "       [-1.02899151,  2.05235723, -1.23887172],\n",
       "       [-1.02899151, -1.39935048,  0.50880188],\n",
       "       [ 0.97182532,  1.06615502, -0.19026756],\n",
       "       [-1.02899151,  0.37581348, -0.53980228],\n",
       "       [-1.02899151, -0.31452806, -0.71456964],\n",
       "       [-1.02899151, -0.80762916,  1.295255  ],\n",
       "       [-1.02899151,  1.36201569,  1.93606866],\n",
       "       [-1.02899151, -0.70900894, -1.64666223],\n",
       "       [-1.02899151, -0.11728762,  1.90694076],\n",
       "       [-1.02899151,  1.46063591,  0.30490663],\n",
       "       [-1.02899151, -0.80762916,  0.21752295],\n",
       "       [-1.02899151,  0.27719326,  0.01362769],\n",
       "       [ 0.97182532, -0.80762916,  0.50880188],\n",
       "       [-1.02899151,  0.9675348 , -1.23887172],\n",
       "       [ 0.97182532,  0.17857304, -0.30677913],\n",
       "       [ 0.97182532, -0.70900894,  0.13013927],\n",
       "       [-1.02899151,  0.67167414, -1.44276698],\n",
       "       [ 0.97182532, -0.90624938,  0.10101137],\n",
       "       [ 0.97182532, -0.0186674 , -0.30677913],\n",
       "       [-1.02899151,  1.36201569,  0.53792977],\n",
       "       [ 0.97182532, -1.20211004, -1.61753434]])"
      ]
     },
     "execution_count": 30,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "X_test_scaled"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 31,
   "id": "3bd2540f-dd90-429c-b999-4e091b7aa5dd",
   "metadata": {},
   "outputs": [],
   "source": [
    "from sklearn.linear_model import LogisticRegression"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 36,
   "id": "22987a99-f80c-4f60-9372-e6f1be069e4b",
   "metadata": {},
   "outputs": [],
   "source": [
    "logistic_reg = LogisticRegression(random_state = 0).fit(X_train_scaled,Y_train) # Training Complete"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 34,
   "id": "1990a1ce-6e05-46b6-a6a1-55831bad9c5f",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0,\n",
       "       1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 1, 1, 0, 0,\n",
       "       0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0,\n",
       "       0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0,\n",
       "       0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 1,\n",
       "       0, 0, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 0,\n",
       "       0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0, 0,\n",
       "       0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0,\n",
       "       1, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 1,\n",
       "       0, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0,\n",
       "       1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0,\n",
       "       0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=int64)"
      ]
     },
     "execution_count": 34,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg.predict(X_train_scaled) "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 35,
   "id": "8a5e8fd3-305d-468d-aba8-068f2fbd1945",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "array([0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0,\n",
       "       0, 0, 0, 1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1,\n",
       "       0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0,\n",
       "       0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0,\n",
       "       0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=int64)"
      ]
     },
     "execution_count": 35,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg.predict(X_test_scaled)"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 37,
   "id": "b47691b2-1574-45fb-a0d1-9436e6e59c72",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8214285714285714"
      ]
     },
     "execution_count": 37,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg.score(X_train_scaled,Y_train)  # Calculating Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 38,
   "id": "b2abeba3-2c03-4ba7-b939-6ab6c2426a58",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0.8214285714285714 means 82.1428..% accurate"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 39,
   "id": "3c8458cc-6d26-427b-9939-39d901bb74b7",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.85"
      ]
     },
     "execution_count": 39,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg.score(X_test_scaled,Y_test)  # Calculating Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 40,
   "id": "4ce2fb9a-72bb-46df-a511-55814b0afc19",
   "metadata": {},
   "outputs": [],
   "source": [
    "# 0.85 = 85% => The model is said to be best the more close accuracy match with 'train' and 'test' "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 41,
   "id": "fcc96ae3-f005-40c0-9305-fc2ef6a56cb8",
   "metadata": {},
   "outputs": [],
   "source": [
    "# We can add some more features for increasing accuracy => "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 61,
   "id": "05edb1f8-6828-4a07-8b1a-9dad0c9d8a13",
   "metadata": {},
   "outputs": [],
   "source": [
    "logistic_reg1 = LogisticRegression(random_state = 0, C = 1, fit_intercept = True).fit(X_train_scaled,Y_train) # Model Trained "
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 45,
   "id": "09b584f4-d523-4103-8a14-bdf6e228e307",
   "metadata": {},
   "outputs": [],
   "source": [
    "# C = regularization parameter, Regularization is a technique used in machine learning to prevent overfitting. Overfitting happens when a model learns the training data too well, including the noise and outliers, which causes it to perform poorly on new data. In simple terms, regularization adds a penalty to the model for being too complex, encouraging it to stay simpler and more general. This way, it’s less likely to make extreme predictions based on the noise in the data and '1' its default value.\n",
    "# fit_intercept = is the intersaction (y = mx + c). True means there should be intersaction."
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 62,
   "id": "44f9ef47-d8f2-4558-adb0-2e95d7d58801",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8214285714285714"
      ]
     },
     "execution_count": 62,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg1.score(X_train_scaled,Y_train)  # Calculating Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 63,
   "id": "e2ab67cb-b8e5-4699-a38b-235768643535",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.85"
      ]
     },
     "execution_count": 63,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg1.score(X_test_scaled,Y_test)  # Calculating Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 67,
   "id": "96e6604f-5efe-49ca-a5ed-80df21941b5d",
   "metadata": {},
   "outputs": [],
   "source": [
    "logistic_reg2 = LogisticRegression(random_state = 0, C = 0.01, fit_intercept = True).fit(X_train_scaled,Y_train) # Model Trained"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 68,
   "id": "6927a9a5-1baf-48cd-9608-db14e7a4813d",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.7607142857142857"
      ]
     },
     "execution_count": 68,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg2.score(X_train_scaled,Y_train)  # Calculating Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 69,
   "id": "cb043133-7f1d-4c5a-8476-25e3298987fa",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8166666666666667"
      ]
     },
     "execution_count": 69,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg2.score(X_test_scaled,Y_test)  # Calculating Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 73,
   "id": "93661a2b-2653-412e-ba5c-0d2deace4f44",
   "metadata": {},
   "outputs": [],
   "source": [
    "logistic_reg3 = LogisticRegression(random_state = 0, C = 50, fit_intercept = True).fit(X_train_scaled,Y_train) # Model Trained"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 74,
   "id": "30f75675-bfcb-4f38-9e43-b7ebd414caaf",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8321428571428572"
      ]
     },
     "execution_count": 74,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg3.score(X_train_scaled,Y_train)  # Calculating Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 75,
   "id": "b552cb1f-e56f-4655-9e93-6dab3eecaeba",
   "metadata": {},
   "outputs": [
    {
     "data": {
      "text/plain": [
       "0.8416666666666667"
      ]
     },
     "execution_count": 75,
     "metadata": {},
     "output_type": "execute_result"
    }
   ],
   "source": [
    "logistic_reg3.score(X_test_scaled,Y_test)  # Calculating Accuracy"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 76,
   "id": "2a79df60-50e4-4bc6-86c2-f3b0ad7f7793",
   "metadata": {},
   "outputs": [],
   "source": [
    "# logistic_reg3 is more accurate then all"
   ]
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.12.3"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
