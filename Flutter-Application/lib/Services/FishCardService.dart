
import 'package:flutter/widgets.dart';
import 'package:app2/FishCard.dart';
import 'package:http/http.dart' as http;
import 'dart:convert' as convert;

import '../constants.dart';

class FishCardService{
  T cast<T>(x) => x is T ? x : null;
  Future<List<FishCard>> getFishes(double latitude,double longitude) async {
    debugPrint("hiii");
    String url = URL+'/getNearbyCatches?latitude='+latitude.toString()+'&longitude='+longitude.toString();
    debugPrint(url);
    var response = await http.get(url);
    var json = convert.jsonDecode(response.body);
    var jsonResults = json['nearestCatches'] as List;
    debugPrint(jsonResults.toList().toString()+"hi");
    List<FishCard> fishCards = [];
    for(var i=0;i<jsonResults.length;i++){
        FishCard f = new FishCard();
        f.latitude = jsonResults[i]['latitude'];
        f.longitude = jsonResults[i]['longitude'];
        f.date = jsonResults[i]['date'];
        f.catchId = jsonResults[i]['catchId'];
        f.weight = jsonResults[i]['totalWeight'];
        List<Catch> cf = [];
        for(var j=0;j<jsonResults[i]['catches'].length;j++){
          cf.add(Catch(cost: jsonResults[i]['catches'][j]['cost'],fishType: jsonResults[i]['catches'][j]['fishType'],family: "B",genus: "a",weight: jsonResults[i]['catches'][j]['weight'],quantity: jsonResults[i]['catches'][j]['quantity']));
        }
        f.catchesFish = cf;
        fishCards.add(f);
    }
    return fishCards;
  }
}