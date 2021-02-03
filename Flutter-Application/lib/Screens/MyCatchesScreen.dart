import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:localstorage/localstorage.dart';
import 'AnalyticsScreen.dart';
import 'Background1.dart';

import 'package:app2/constants.dart';
import 'package:localstorage/localstorage.dart';
// ignore: unused_import
import 'RecordCatchScreen.dart';
import 'package:geolocator/geolocator.dart';
import 'UpdatesScreen.dart';
import 'ExploreScreen.dart';
class MyCatchesScreen extends StatelessWidget {
  @override
  Widget build(BuildContext context) {

    return MaterialApp(
      title: 'My Catches',
      debugShowCheckedModeBanner: false,
      theme: _buildShrineTheme(),
      home: CatchList(title: 'List of My Catches'),
      routes: {
        '/updates':(context)=> UpdatesScreen(),
        '/record':(context)=> RecordCatchScreen(),
        '/explore':(context)=> ExploreScreen(),
        '/analytics':(context)=> AnalyticsScreen()
      },
    );
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
  final String apiUrl = URL+"/getCatchesHistory?mobile=";
  List<dynamic> _catches = [];
  final LocalStorage storage = new LocalStorage('localstorage_app');
  Geolocator _geolocator;


  void fetchCatches() async {
    final LocalStorage storage = new LocalStorage('localstorage_app');
    var mobile = storage.getItem('number').toString();
    var result = await http.get(apiUrl+mobile);
    setState(() {
      _catches = json.decode(result.body)['catches'];
      debugPrint(_catches.toList().toString());
    });
  }


  Widget _buildList() {
    return _catches.length != 0
        ? RefreshIndicator(
            child: ListView.builder(
                padding: EdgeInsets.all(12.0),
                itemCount: _catches.length,
                itemBuilder: (BuildContext context, int index) {
                  return Card(
              child: Column(
                children: [
                  ListTile(
                    leading: Icon(Icons.card_giftcard),
                    title: Text(_catches[index]['name'],style: TextStyle(fontSize: 20),),
                    subtitle: Text(
                      _catches[index]['date'],
                      style: TextStyle(color: Colors.black.withOpacity(0.6)),
                    ),
                  ),
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Text(
                      _catches[index]['description'] + ' at the location with latitude' + _catches[index]['latitude'].toString() + ' and longitude '+ _catches[index]['longitude'].toString() + "\n\n The weight of the catch was : "+ _catches[index]['weight'].toString(),
                      style: TextStyle(color: Colors.black.withOpacity(0.6),fontSize: 18),
                    ),
                  ),
                  (_catches[index]['image'] == null) ? Image.asset('assets/images/fish.png') : Image.network(_catches[index]['image']),
                  SizedBox(height:20),
                ],
              ),
            );
                }),
            onRefresh: _getData,
          )
        : Center(child: CircularProgressIndicator());
  }

  Future<void> _getData() async {
    setState(() {
      fetchCatches();
    });
  }

    void checkPermission() {
    _geolocator.checkGeolocationPermissionStatus().then((status) {
      print('status: $status');
    });
    _geolocator
        .checkGeolocationPermissionStatus(
            locationPermission: GeolocationPermission.locationAlways)
        .then((status) {
      print('always status: $status');
    });
    _geolocator.checkGeolocationPermissionStatus(
        locationPermission: GeolocationPermission.locationWhenInUse)
      ..then((status) {
        print('whenInUse status: $status');
      });
  }



  @override
  void initState() {
    
    super.initState();
    _geolocator = Geolocator();
    checkPermission();
    fetchCatches();
  }

  @override
  Widget build(BuildContext context) {
    return Background1(
          child: Scaffold(
        appBar: AppBar(
            title: const Text('Your Catches')),
            drawer: navigationDrawer(),
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

// ignore: camel_case_types
class navigationDrawer extends StatelessWidget {
  @override
  Widget build(BuildContext context) {
    return Drawer(
        child: Container(
      color: shrinePink100,
      child: ListView(
        children: <Widget>[
          createDrawerHeader(),
          createDrawerBodyItem(
            icon: Icons.update,
            text: 'Marine Weather Updates',
            onTap: () => Navigator.pushNamed(context, '/updates'),
          ),
          createDrawerBodyItem(
            icon: Icons.record_voice_over,
            text: 'Record Catch',
            onTap: () => Navigator.pushNamed(context, '/record'),
          ),
          createDrawerBodyItem(
            icon: Icons.explore,
            text: 'Explore Catches',
            onTap: () => Navigator.pushNamed(context, '/explore'),
          ),
          createDrawerBodyItem(
            icon: Icons.analytics,
            text: 'Analytics',
            onTap: () => Navigator.pushNamed(context, '/analytics'),
          ),
          ListTile(
            title: Text('2.0.0', style: TextStyle(color: Colors.black)),
            onTap: () {},
          ),
        ],
      ),
    ));
  }
}

Widget createDrawerBodyItem(
    {IconData icon, String text, GestureTapCallback onTap}) {
  return ListTile(
    title: Row(
      children: <Widget>[
        Icon(
          icon,
          color: Colors.blue,
        ),
        Padding(
          padding: EdgeInsets.only(left: 18.0),
          child: Text(text,
              style: TextStyle(
                  color: Colors.black,
                  fontSize: 16,
                  fontWeight: FontWeight.w400)),
        ),
      ],
    ),
    onTap: onTap,
  );
}

Widget createDrawerHeader() {
  return DrawerHeader(
    child: Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: <Widget>[
        Text(
          'Your Dashboard',
          style: TextStyle(
              color: Colors.black, fontSize: 16.0, fontWeight: FontWeight.w600),
        ),
      ],
    ),
  );
}