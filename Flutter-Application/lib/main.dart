import 'package:app2/Screens/Welcome/welcome_screen.dart';
import 'package:app2/constants.dart';
import 'package:flutter/material.dart';
import 'package:app2/Screens/ExploreScreen.dart';
import 'package:intro_slider/intro_slider.dart';
import 'package:intro_slider/slide_object.dart';
void main() {
  runApp(BasicApp());
}

class BasicApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Jan Dhan Drashak',
      theme: ThemeData(primarySwatch: Colors.blue),
      home: IntroScreen(),
    );
  }
}
class IntroScreen extends StatefulWidget {
  @override
  _IntroScreenState createState() => _IntroScreenState();
}

class _IntroScreenState extends State<IntroScreen> {
  List<Slide> slides = new List();

  @override
  void initState() {
    super.initState();

    slides.add(
      new Slide(
        title: "FISH-IT",
        styleTitle: TextStyle(
            color: Colors.black, fontSize: 25, fontFamily: 'Quicksand'),
        description:
            "One stop solution for all your fishing needs",
        styleDescription: TextStyle(
            color: Colors.black, fontSize: 14, fontFamily: 'Quicksand'),
        pathImage: "assets/images/fish.png",
        heightImage: 125.0,
        widthImage: 100.0,
        backgroundColor: const Color(0xf2f8f9ff),
      ),
    );

    slides.add(
      new Slide(
        title: "Record and View Your Catches",
        styleTitle: TextStyle(
            color: Colors.black, fontSize: 25, fontFamily: 'Quicksand'),
        description:
            "One stop solution to record catches and view your catch history",
        styleDescription: TextStyle(
            color: Colors.black, fontSize: 14, fontFamily: 'Quicksand'),
        pathImage: "assets/images/fish.png",
        heightImage: 125.0,
        widthImage: 100.0,
        backgroundColor: const Color(0xf2f8f9ff),
      ),
    );
    slides.add(
      new Slide(
        title: "Realtime Marine Updates",
        styleTitle: TextStyle(
            color: Colors.black, fontSize: 22, fontFamily: 'Quicksand'),
        description:
            "Get Real time updates right from the Ministry of Weather and Horticulture",
        styleDescription: TextStyle(
            color: Colors.black, fontSize: 14, fontFamily: 'Quicksand'),
        pathImage: "assets/images/fish.png",
        heightImage: 125.0,
        widthImage: 100.0,
        backgroundColor: const Color(0xf2f8f9ff),
      ),
    );
    slides.add(
      new Slide(
        title: "Explore Nearby Catches",
        styleTitle: TextStyle(
            color: Colors.black, fontSize: 22, fontFamily: 'Quicksand'),
        description:
            "Explore High value fishes catched near you.",
        styleDescription: TextStyle(
            color: Colors.black, fontSize: 14, fontFamily: 'Quicksand'),
        pathImage: "assets/images/fish.png",
        heightImage: 125.0,
        widthImage: 100.0,
        backgroundColor: const Color(0xf2f8f9ff),
      ),
    );
  }


  void onDonePress() {
    // TODO: go to next screen
    Navigator.of(context)
        .push(MaterialPageRoute(builder: (context) => MyApp()));
  }

  void onSkipPress() {
    // TODO: go to next screen
    onDonePress();
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
        body: IntroSlider(
            slides: this.slides,
            onDonePress: this.onDonePress,
            onSkipPress: this.onSkipPress,
            nameDoneBtn: 'Done',
            nameNextBtn: 'Next',
            nameSkipBtn: 'Skip',
            styleNameDoneBtn:
                TextStyle(color: Colors.blue[600], fontSize: 16.0),
            styleNamePrevBtn:
                TextStyle(color: Colors.blue[600], fontSize: 16.0),
            styleNameSkipBtn:
                TextStyle(color: Colors.blue[600], fontSize: 16.0),
            colorActiveDot: Colors.blue[600],
            colorDot: Colors.grey));
  }
}


class MyApp extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Fisherman App',
      theme: ThemeData(
        primaryColor: kPrimaryColor,
        scaffoldBackgroundColor: Colors.white
      ),
      home: WelcomeScreen(),
    );
  }
}
