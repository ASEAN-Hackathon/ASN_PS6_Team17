class Catch{
  int cost;
  String fishType;
  String family;
  String genus;
  var weight;
  int quantity;

  Catch({
    this.cost,
    this.fishType,
    this.family,
    this.genus,
    this.weight,
    this.quantity
  }); 
}

class FishCard {
  double latitude;
  double longitude;
  int catchId;
  String date;
  var weight;
  List<Catch> catchesFish = [];

  FishCard({
    this.latitude,
    this.longitude,
    this.catchesFish
    });

}