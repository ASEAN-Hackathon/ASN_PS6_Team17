import 'dart:async';
import 'dart:convert';
import 'dart:io';
import 'package:app2/Screens/MyCatchesScreen.dart';
import 'package:image_picker/image_picker.dart';
import 'package:flutter/material.dart';
import 'package:progress_dialog/progress_dialog.dart';
import 'package:http/http.dart' as http;
import 'package:image_cropper/image_cropper.dart';
import '../styles/styles.dart';
import 'package:app2/constants.dart';
import 'package:localstorage/localstorage.dart';


class RecordCatchScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MyApp1();
  }
}

class MyApp1 extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: Text('Record Catch')), body: MyCustomForm());
  }
}

class MyCustomForm extends StatefulWidget {
  @override
  MyCustomFormState createState() {
    return MyCustomFormState();
  }
}

enum AppState {
  free,
  picked,
  cropped,
}

class MyCustomFormState extends State<MyCustomForm> {
  final LocalStorage storage = new LocalStorage('localstorage_app');
  final apiKey = "AIzaSyA1Pj-1B1e6Pf29c7Y1FuIA7WNdhncQw-E";
  AppState state1;
  File _catchImage;

  final _formKey = GlobalKey<FormState>();
  final nameController = TextEditingController();
  final descriptionController = TextEditingController();
  final dateController = TextEditingController();
  final weightController = TextEditingController();
  final hoursController = TextEditingController();
  final latitudeController = TextEditingController();
  final longitudeController = TextEditingController();

  ProgressDialog pr;
  void _showDialog(var dialog1, var dialog2) {
    showDialog(
      context: context,
      builder: (BuildContext context) {
        return AlertDialog(
          title: new Text(dialog1),
          content: new Text(
              dialog2),
          actions: <Widget>[
            new FlatButton(
              child: new Text("OK"),
              onPressed: () {
                Navigator.push(
                  context,
                  MaterialPageRoute(builder: (context) => MyCatchesScreen()),
                );
              },
            ),
          ],
        );
      },
    );
  }


  Future uploadData() async {
          await pr.show();
          if (_catchImage == null) return;
          String base64Image = base64Encode(_catchImage.readAsBytesSync());
          String fileName = _catchImage.path.split("/").last;

          final uri = URL+"/updateCatch";
              Map requestBody = {
                  "image": base64Image,
                  "imageFileName":fileName,
                  "name":nameController.text.toString(),
                  "description":descriptionController.text.toString(),
                  "date":dateController.text.toString(),
                  "weight":weightController.text.toString(),
                  "latitude":latitudeController.text.toString(),
                  "longitude":longitudeController.text.toString(),
                  "number":storage.getItem('number').toString(),
                  'hours':hoursController.text.toString()
              };
              String body = json.encode(requestBody);
              print(requestBody);
              http.Response response = await http.post(
                  uri,
                  headers: {"Content-Type": "application/json"},
                  body: body,
              );
              final responseBody = json.decode(response.body);
              if(responseBody['success'] == true){
                    pr.hide().whenComplete(() {
                    _showDialog("Catch Details Saved","Thank you for updating the catch. It has been saved successfully.");
                  });
              }
              else{
                  pr.hide().whenComplete(() {
                    _showDialog("Some error occured","There was an issue uploading the catch");
                  });
              }
  }

  Future<Null> _pickImage1() async {
    _catchImage = await ImagePicker.pickImage(source: ImageSource.gallery);
    if (_catchImage != null) {
      setState(() {
        state1 = AppState.picked;
      });
    }
  }

  Future<Null> _cropImage1() async {
    File croppedFile = await ImageCropper.cropImage(
        sourcePath: _catchImage.path,
        aspectRatioPresets: Platform.isAndroid
            ? [
                CropAspectRatioPreset.square,
                CropAspectRatioPreset.ratio3x2,
                CropAspectRatioPreset.original,
                CropAspectRatioPreset.ratio4x3,
                CropAspectRatioPreset.ratio16x9
              ]
            : [
                CropAspectRatioPreset.original,
                CropAspectRatioPreset.square,
                CropAspectRatioPreset.ratio3x2,
                CropAspectRatioPreset.ratio4x3,
                CropAspectRatioPreset.ratio5x3,
                CropAspectRatioPreset.ratio5x4,
                CropAspectRatioPreset.ratio7x5,
                CropAspectRatioPreset.ratio16x9
              ],
        androidUiSettings: AndroidUiSettings(
            toolbarTitle: 'Cropper',
            toolbarColor: Colors.blue[600],
            toolbarWidgetColor: Colors.white,
            initAspectRatio: CropAspectRatioPreset.original,
            lockAspectRatio: false),
        iosUiSettings: IOSUiSettings(
          title: 'Cropper',
        ));
    if (croppedFile != null) {
      _catchImage = croppedFile;
      setState(() {
        state1 = AppState.cropped;
      });
    }
  }

  void _clearImage1() {
    _catchImage = null;
    setState(() {
      state1 = AppState.free;
    });
  }

  @override
  void initState() {
    super.initState();
    state1 = AppState.free;
  }


  @override
  Widget build(BuildContext context) {
    pr = ProgressDialog(context,
        type: ProgressDialogType.Normal, isDismissible: false, showLogs: false);
    pr.style(
        message: '\tUploading Data...',
        borderRadius: 10.0,
        backgroundColor: Colors.white,
        progressWidget:
            SizedBox(height: 50, width: 50, child: CircularProgressIndicator()),
        elevation: 10.0,
        insetAnimCurve: Curves.easeInOut,
        progressTextStyle: TextStyle(
            color: Colors.black, fontSize: 15.0, fontWeight: FontWeight.bold),
        messageTextStyle: TextStyle(
            color: Colors.black, fontSize: 15.0, fontWeight: FontWeight.bold));
    return SingleChildScrollView(
        child: Form(
            key: _formKey,
            child: Container(
              width: MediaQuery.of(context).size.width - 9.0,
              padding: EdgeInsets.all(25.0),
              child: Column(
                children: <Widget>[
                  TextFormField(
                      controller: nameController,
                      decoration: Style.inputDecor(
                          Icon(Icons.call_to_action_sharp), 'Name', 'Enter Name of catch')),
                  Style.space(),
                  TextFormField(
                    controller: dateController,
                    decoration: Style.inputDecor(Icon(Icons.calendar_today),
                        'Select Date', 'Date of the Catch'),
                    keyboardType: TextInputType.text,
                  ),
                  Style.space(),
                  TextFormField(
                      controller: descriptionController,
                      decoration: Style.inputDecor(Icon(Icons.description),
                          'Description', 'Enter the Description'),
                      keyboardType: TextInputType.multiline),
                  Style.space(),

                  TextFormField(
                      controller: weightController,
                      decoration: Style.inputDecor(
                          Icon(Icons.line_weight),
                          'Weight',
                          'Enter weight of catch'),
                      keyboardType: TextInputType.number),
                  Style.space(),
                  TextFormField(
                      controller: hoursController,
                      decoration: Style.inputDecor(
                          Icon(Icons.hourglass_bottom),
                          'Hours Spent',
                          'Hours Spent on the catch.'),
                      keyboardType: TextInputType.number),
                  Style.space(),
                  TextFormField(
                      controller: latitudeController,
                      decoration: Style.inputDecor(
                          Icon(Icons.line_weight),
                          'Latitude',
                          'Enter latitude'),
                      keyboardType: TextInputType.number),
                  Style.space(),
                  TextFormField(
                      controller: longitudeController,
                      decoration: Style.inputDecor(
                          Icon(Icons.line_weight),
                          'Longitude',
                          'Enter longitude'),
                      keyboardType: TextInputType.number),
                  Style.space(),
                  

                  // RaisedButton(
                  //           color: Colors.blue[100],
                  //           onPressed: () async {
                  //             LocationResult result = await showLocationPicker(
                  //               context,
                  //               apiKey,
                  //               initialCenter: LatLng(21,22),
                  //               automaticallyAnimateToCurrentLocation: true,
                  //               myLocationButtonEnabled: true,
                  //               layersButtonEnabled: true,
                  //               resultCardAlignment: Alignment.bottomCenter,
                  //             );
                  //             print("result = $result");
                  //             setState(() => _pickedLocation = result);
                  //           },
                  //           child: Container(
                  //               height: 49.0,
                  //               width: 200.0,
                  //               decoration: BoxDecoration(
                  //                   borderRadius: BorderRadius.all(
                  //                       Radius.circular(10.0))),
                  //               child: Row(
                  //                   crossAxisAlignment:
                  //                       CrossAxisAlignment.center,
                  //                   mainAxisSize: MainAxisSize.min,
                  //                   children: [
                  //                     Icon(
                  //                       Icons.edit_location,
                  //                       color: Colors.blue[600],
                  //                     ),
                  //                     Text(
                  //                       'Pick Location',
                  //                       style:
                  //                           TextStyle(color: Colors.blue[600]),
                  //                     ),
                  //                   ]))),
                  Style.space(),
                  Column(
                    children: <Widget>[
                      Container(
                        height: 200,
                        child: (_catchImage != null)
                            ? Image.file(
                                _catchImage,
                                fit: BoxFit.fill,
                              )
                            : GestureDetector(
                                onTap: () => (state1 == AppState.free)
                                    ? _pickImage1()
                                    : (state1 == AppState.picked)
                                        ? _cropImage1()
                                        : _clearImage1(),
                                child:
                                    Style.greyBox(context,'Catch Image')),
                      ),
                      Style.space(),
                    ],
                  ),
                  Style.space(),
                  Container(
                    width: MediaQuery.of(context).size.width,
                    child: MaterialButton(
                      height: 49.0,
                      color: Theme.of(context).primaryColor,
                      padding: EdgeInsets.all(10),
                      textColor: Colors.white,
                      child: new Text(
                        'submit',
                        style: TextStyle(fontSize: 18.0),
                      ),
                      onPressed: () => {uploadData()},
                      splashColor: Colors.redAccent,
                    ),
                  )
                ],
              ),
            )));
  }
}