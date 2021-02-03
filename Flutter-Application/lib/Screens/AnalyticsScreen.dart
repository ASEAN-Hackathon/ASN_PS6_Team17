import 'dart:async';
import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'package:localstorage/localstorage.dart';
import '../constants.dart';
import 'Background1.dart';



class AnalyticsScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return MyApp1();
  }
}

class MyApp1 extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Scaffold(
        appBar: AppBar(title: Text('Analytics for your catches')), body: CatchList());
  }
}

class CatchList extends StatefulWidget {

  final String title;
  CatchList({Key key, this.title}) : super(key: key);


  @override
  State<StatefulWidget> createState() {
    return _CatchListState();
  }
}

class _CatchListState extends State<CatchList> {
  final String apiUrl = URL+"/monthVsCpueForFisherman?number=";
  final LocalStorage storage = new LocalStorage('localstorage_app');
  double average ;
  var url;
  double moneySaved;
  void fetchCatches() async {
    final LocalStorage storage = new LocalStorage('localstorage_app');
    var mobile = storage.getItem('number').toString();
    var result = await http.get(apiUrl+mobile);
    setState(() {
      url = json.decode(result.body)['chartUrl'];
      average = json.decode(result.body)['average'];
      moneySaved = json.decode(result.body)['moneySaved'];
    });
  }


  Widget _buildList() {
    return (url != null && average != null && moneySaved != null)
        ? 
          ListView.builder(
            padding: EdgeInsets.all(12.0),
            itemCount: 1,
            itemBuilder: (BuildContext context, int index) {
              return Card(
                color: shrinePink100,
                child: Column(
                  children: [
                    ListTile(
                      leading: Icon(Icons.info),
                      title: Text("Your Average CPUE is: "+ average.toString(),style: TextStyle(fontSize: 20,),),
                      subtitle: Text(
                        "\n\nYou saved Rs "+ moneySaved.toString() +" with the application",
                        style: TextStyle(color: Colors.black.withOpacity(0.6),fontSize: 16),
                      ),
                    ),
                    (url == null) ? Image.asset('assets/images/fish.png') : Image.network(url,height: 500,),
                    SizedBox(height:20),
                  ],
                ),
              );
            })
        : Center(child: CircularProgressIndicator());
  }

  Future<void> _getData() async {
    setState(() {
      fetchCatches();
    });
  }




  @override
  void initState() {
    
    super.initState();
    fetchCatches();
  }

  @override
  Widget build(BuildContext context) {
    return Background1(
          child: Scaffold(
            body: Container(
              child: _buildList(),
            ),
          ),
    );
  }
}



IconThemeData _customIconTheme(IconThemeData original) {
  return original.copyWith(color: shrineBrown900);
}

ThemeData _buildShrineTheme() {
  final ThemeData base = ThemeData.light();
  return base.copyWith(
    colorScheme: _shrineColorScheme,
    accentColor: shrineBrown900,
    primaryColor: shrinePink100,
    buttonColor: shrinePink100,
    scaffoldBackgroundColor: shrineBackgroundWhite,
    cardColor: shrineBackgroundWhite,
    textSelectionColor: shrinePink100,
    errorColor: shrineErrorRed,
    buttonTheme: const ButtonThemeData(
      colorScheme: _shrineColorScheme,
      textTheme: ButtonTextTheme.normal,
    ),
    primaryIconTheme: _customIconTheme(base.iconTheme),
    textTheme: _buildShrineTextTheme(base.textTheme),
    primaryTextTheme: _buildShrineTextTheme(base.primaryTextTheme),
    accentTextTheme: _buildShrineTextTheme(base.accentTextTheme),
    iconTheme: _customIconTheme(base.iconTheme),
  );
}

TextTheme _buildShrineTextTheme(TextTheme base) {
  return base
      .copyWith(
    headline: base.headline.copyWith(
      fontWeight: FontWeight.w500,
      letterSpacing: defaultLetterSpacing,
    ),
    title: base.title.copyWith(
      fontSize: 18,
      letterSpacing: defaultLetterSpacing,
    ),
    caption: base.caption.copyWith(
      fontWeight: FontWeight.w400,
      fontSize: 14,
      letterSpacing: defaultLetterSpacing,
    ),
    body2: base.body2.copyWith(
      fontWeight: FontWeight.w500,
      fontSize: 16,
      letterSpacing: defaultLetterSpacing,
    ),
    body1: base.body1.copyWith(
      letterSpacing: defaultLetterSpacing,
    ),
    subhead: base.subhead.copyWith(
      letterSpacing: defaultLetterSpacing,
    ),
    display1: base.display1.copyWith(
      letterSpacing: defaultLetterSpacing,
    ),
    button: base.button.copyWith(
      fontWeight: FontWeight.w500,
      fontSize: 14,
      letterSpacing: defaultLetterSpacing,
    ),
  )
      .apply(
    fontFamily: 'Rubik',
    displayColor: shrineBrown900,
    bodyColor: shrineBrown900,
  );
}

const ColorScheme _shrineColorScheme = ColorScheme(
  primary: shrinePink100,
  primaryVariant: shrineBrown900,
  secondary: shrinePink50,
  secondaryVariant: shrineBrown900,
  surface: shrineSurfaceWhite,
  background: shrineBackgroundWhite,
  error: shrineErrorRed,
  onPrimary: shrineBrown900,
  onSecondary: shrineBrown900,
  onSurface: shrineBrown900,
  onBackground: shrineBrown900,
  onError: shrineSurfaceWhite,
  brightness: Brightness.light,
);

const Color shrinePink50 = Color(0xFFFEEAE6);
const Color shrinePink100 = Color(0xFFFEDBD0);
const Color shrinePink300 = Color(0xFFFBB8AC);
const Color shrinePink400 = Color(0xFFEAA4A4);

const Color shrineBrown900 = Color(0xFF442B2D);
const Color shrineBrown600 = Color(0xFF7D4F52);

const Color shrineErrorRed = Color(0xFFC5032B);

const Color shrineSurfaceWhite = Color(0xFFFFFBFA);
const Color shrineBackgroundWhite = Colors.white;

const defaultLetterSpacing = 0.03;
