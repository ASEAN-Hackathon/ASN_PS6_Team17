import 'dart:convert';

import 'package:app2/Screens/Signup/signup_screen.dart';
import 'package:flutter/material.dart';
import 'package:app2/Screens/Login/login_screen.dart';
import 'package:app2/Screens/Signup/components/background.dart';
import 'package:app2/components/already_have_an_account_acheck.dart';
import 'package:app2/components/rounded_button.dart';
import 'package:app2/components/rounded_input_field.dart';
import 'package:app2/components/rounded_password_field.dart';
import 'dart:math';
import 'package:http/http.dart' as http;

import 'package:awesome_dialog/awesome_dialog.dart';

import '../../../constants.dart';

//ignore: must_be_immutable
class Body extends StatelessWidget {

  var name,govRegId,email,password,mobileNo;

  int getFishermanId() {
    var rnd = new Random();
    var next = rnd.nextDouble() * 1000000;
    while (next < 100000) {
      next *= 10;
    }
    return next.toInt();
  }

  // ignore: non_constant_identifier_names
  void show_success_dialog(BuildContext context){
    print("hello");
    AwesomeDialog(
            context: context,
            animType: AnimType.LEFTSLIDE,
            headerAnimationLoop: false,
            dialogType: DialogType.SUCCES,
            title: 'Success',
            desc:
                'Signup Success, Hit Ok, to log in',
            btnOkOnPress: () {
              debugPrint("hello from success");
              Navigator.push(context,MaterialPageRoute(builder: (context)=> LoginScreen()));
            },
            btnOkIcon: Icons.check_circle,
            onDissmissCallback: () {
              debugPrint('Dialog Dissmiss from callback');
            }).show();
  }

  // ignore: non_constant_identifier_names
  void show_error_dialog(BuildContext context){
    
        AwesomeDialog(
            context: context,
            dialogType: DialogType.ERROR,
            animType: AnimType.RIGHSLIDE,
            headerAnimationLoop: false,
            title: 'Error',
            desc:
                'Some error occured. Please try again.',
            btnOkOnPress: () {
              Navigator.push(context,MaterialPageRoute(builder: (context)=> SignUpScreen()));
            },
            btnOkIcon: Icons.cancel,
            btnOkColor: Colors.red).show();
  }

  @override
  Widget build(BuildContext context) {
    Size size = MediaQuery.of(context).size;
    return Background(
      child: SingleChildScrollView(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: <Widget>[
            Text(
              "SIGNUP",
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            SizedBox(height: size.height * 0.03),
            Image.asset(
              "assets/images/fish.png",
              height: size.height * 0.35,
            ),
            RoundedInputField(
              hintText: "Your Name",
              onChanged: (value) {
                this.name = value;
              },
            ),
            RoundedInputField(
              hintText: "Your GovtId Number",
              onChanged: (value){
                this.govRegId = value;
              }
            ),
            RoundedInputField(
              hintText: "Your Mobile Number",
              onChanged: (value) {
                this.mobileNo = value;
              },
            ),
            RoundedInputField(
              hintText: "Your Email",
              onChanged: (value) {
                this.email = value;
              },
            ),
            RoundedPasswordField(
              onChanged: (value) {
                this.password = value;
              },
            ),
            RoundedButton(
              text: "SIGNUP",
              press: () async{
                final uri = URL+"/signup";
              Map requestBody = {
                "fisherManName":name,
                "fisherManId":getFishermanId(),
                "fisherManGovRegsNumber": int.parse(govRegId),
                "password":password,
                "email":email,
                "mobileNo":int.parse(mobileNo)
              };

              String body = json.encode(requestBody);
              print(requestBody);
              http.Response response = await http.post(
                  uri,
                  headers: {"Content-Type": "application/json"},
                  body: body,
              );
              final responseBody = json.decode(response.body);
              if(responseBody["success"] == true){
                  print("hi");
                  show_success_dialog(context);
              }
              else{
                  show_error_dialog(context);
              }
              },
            ),
            SizedBox(height: size.height * 0.03),
            AlreadyHaveAnAccountCheck(
              login: false,
              press: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(
                    builder: (context) {
                      return LoginScreen();
                    },
                  ),
                );
              },
            ),
          ],
        ),
      ),
    );
  }
}
