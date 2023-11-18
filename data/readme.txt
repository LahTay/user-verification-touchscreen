This script will download data from https://drive.google.com/drive/u/1/folders/1y2VBNGt5vQzsxQsWKjFaukaUaOrY4azH.
I will also rename them so that they conform our number_pattern standard.
Running this script again will download only the files that weren't downloaded previously
if you want to download every file again remove the processed_files.txt

There are steps to be done before you can run this script.
You have to download your secrets.json file and place it in the CURRENT directory

How to download the secrets file:

1.
Go to the Google Developers Console: https://console.developers.google.com/
a) Click on Select a project and choose New Project.
b) Enter project name and create it.

2.
Enable Google Drive API
a) Press Enable Apis and Services
b) Search for: Google Drive API and select it
c) Click: Enable

3.
Configure OAuth
a) In the sidebar under "APIs & Services", select "OAuth consent screen".
b) Select user type (usually external) and press create.
c) Fill necessary name and emails and press continue
d) In "Scopes" just press continue
e) In "test users" add YOUR Google email as a test user (it should be visible below) and press continue.

4.
Create Credentials
a) In the sidebar under "APIs & Services", select "Credentials".
b) Click "Create Credentials" at the top of the page and choose "OAuth client ID"
c) Select "Desktop app" as the Application type.
d) Name your OAuth 2.0 client and click "Create".
e) After the OAuth client is created, click "OK".

5.
Step 5: Download client_secrets.json
a) In the "Credentials" page, under "OAuth 2.0 Client IDs",
 you'll see the client you just created.
b) Click the download icon (it looks like a down arrow)
 to the right of your OAuth 2.0 client.
c) This will download the client_secrets.json file to your computer.

6. Put the secret file in the data folder
REMEMBER TO NOT ADD IT TO GIT
a) Rename it to "client_secrets.json"
b) After running download_data.py you will be asked to log in)

You are finished