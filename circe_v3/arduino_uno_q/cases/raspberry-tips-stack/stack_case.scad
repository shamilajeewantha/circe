// stack_case.scad — Gehaeuse fuer Arduino UNO Q + UNO Media Carrier
// Spaghetti-Waechter / raspberry.tips — v0.3 (24.08.2026)
//
// Teile: openscad -D part=\"boden\" -o boden.stl stack_case.scad
//   boden | haube | wall_tab | preview
//
// v0.3 — ARCHITEKTURWECHSEL nach Philipps Einwand: Buchsen stehen LINKS
// UND RECHTS 2 mm ueber die Boardkante → der Stack passt nicht von oben
// in eine Wanne mit geschlossenen Fenstern. Deshalb Stuelp-Prinzip:
//   * BODEN: flache Platte mit 4 Eck-Auflagen (9 mm) + Rand-Lippe.
//     Stack wird einfach draufgelegt.
//   * HAUBE: Deckel MIT Waenden, stuelpt von oben drueber. Alle Port-
//     Ausschnitte sind zur Haubenunterkante OFFEN und gleiten beim
//     Aufsetzen ueber Buchsen/USB-C. 4 Niederhalter-Saeulen spannen den
//     Stack. 2x M3 seitlich in die Bodenlippe.
//   * Druck: beide Teile flach, KEIN Support (offene Ausschnitte zeigen
//     in Druckorientierung nach oben).
//
// Gemessen 24.08. (Referenz = Unterkante Carrier-Platine):
//   Stack UK Carrier→OK UNO Q 10.0 · USB-C ab OK UNO Q (~22..25 abs),
//   2 mm Kantenueberstand · Klinken+CSI UNTEN am Carrier, Buchsenmitte
//   −3 mm, je 2 mm Kantenueberstand beidseitig · Carrier rechts +3 mm.
// Offen: A2 (USB-Position laengs), A3 (welche Wand Klinken/CSI —
// Fenster beidseitig), A4 (LED-Fenster), Eckfreiheit unterm Carrier.

include <lib_gopro.scad>;

part = "preview";
$fn = $preview ? 32 : 64;

// ── Basis ─────────────────────────────────────────────────────
brd  = [68.58, 53.34];
holes = [[13.97, 2.54], [15.24, 50.80], [66.04, 7.62], [66.04, 35.56]];
clr = 0.6; wall = 2.4; tray_t = 3.0;
carrier_ext = 3.0;
sup_h = 8.0; stack_h = 10.0; comp_top = 8.5; top_clr = 3.0;  // 24.08.: Auflagen -1, Haube +1

rim_t = 2.0; rim_h = 5.0; rim_gap = 0.15;  // 24.08.: Spiel halbiert

cav = [brd[0] + 2*clr + carrier_ext, brd[1] + 2*clr];   // 72.78 x 54.54
m   = wall + rim_gap + rim_t;                            // 4.7 Rand → Kavitaet
outerd = [cav[0] + 2*m, cav[1] + 2*m];                   // 82.18 x 63.94
bx = m + clr; by = m + clr;                              // Board-Nullpunkt (UNO Q links)

z_carrier = tray_t + sup_h;              // 12
z_unoq    = z_carrier + stack_h;         // 22
z_in_top  = z_unoq + comp_top + top_clr; // 32.5 Innenhoehe
haube_bot = tray_t + 0.2;                // Wand-Unterkante (0.2 Fuge)
H         = z_in_top + wall;             // 34.9 Gesamt

// Fensterspannen (Board-Koordinaten, grosszuegig — A2/A3 offen)
win_x  = [9, 57];         // Klinken/CSI-Zone: +8 zur USB-Seite (24.08., endgueltig)
usb_y  = [25, 45];        // USB-C links, Mitte y~35 (FOTO 24.08.)
qwiic_y = [10.8, 22];     // Qwiic-Fenster hinter dem Pfostenrand (Saeule endet y 10.42)

// ── BODEN: Platte + Lippe + Eck-Auflagen ──────────────────────
module boden() {
    // Platte
    cube([outerd[0], outerd[1], tray_t]);
    // Rand-Lippe (Ring), an den Fensterzonen unterbrochen, damit die
    // haengenden Buchsen (ab abs 5.5) frei bleiben
    difference() {
        translate([m - rim_t, m - rim_t, tray_t])
            cube([cav[0] + 2*rim_t, cav[1] + 2*rim_t, rim_h]);
        translate([m, m, tray_t - 0.5])
            cube([cav[0], cav[1], rim_h + 1]);
        // Unterbrechungen an den Laengsseiten (Fensterzonen)
        for (y = [m - rim_t - 0.5, m + cav[1] - 0.5])
            translate([bx + win_x[0], y, tray_t - 0.5])
                cube([win_x[1] - win_x[0], rim_t + 1, rim_h + 1]);
        // M3-Pilotbohrungen fuer die Haubenschrauben (x = bx+62)
        for (y = [m - rim_t - 0.5, m + cav[1] - 0.5])
            translate([bx + 62, y - 0.2, tray_t + 2.5])
                rotate([-90, 0, 0]) cylinder(d = 2.5, h = rim_t + 1.5);
    }
    // Schnapp-Nocken, Diamantprofil, auf den STIRNSEITEN (dort ist Platz:
    // links frei vom USB-Fenster, rechts frei von DSI/Qwiic — 24.08.)
    for (yr = [5, 50])
        translate([m - rim_t, by + yr, tray_t + rim_h/2 + 0.5])
            rotate([0, 45, 0]) cube([1.1, 4, 1.1], center = true);
    for (yr = [4, 50])
        translate([outerd[0] - (m - rim_t), by + yr, tray_t + rim_h/2 + 0.5])
            rotate([0, 45, 0]) cube([1.1, 4, 1.1], center = true);

    // 4 Eck-Auflagen fuer den Carrier
    pad = 9;
    px = [m, m + cav[0] - pad];
    py = [m, m + cav[1] - pad];
    for (i = [0, 1], j = [0, 1])
        translate([px[i], py[j], tray_t]) cube([pad, pad, sup_h]);
}

// ── HAUBE: Deckel mit Waenden, Ausschnitte nach unten offen ───
module haube() {
    difference() {
        union() {
            // Waende
            difference() {
                translate([0, 0, haube_bot]) cube([outerd[0], outerd[1], H - haube_bot]);
                translate([wall, wall, haube_bot - 0.5])
                    cube([outerd[0] - 2*wall, outerd[1] - 2*wall, H - haube_bot - wall + 0.5]);
            }
            // Deckplatte
            translate([0, 0, z_in_top]) cube([outerd[0], outerd[1], wall]);
        }

        // Klinken/CSI: beide Laengswaende, UNTEN OFFEN bis abs 14
        for (y = [-0.5, outerd[1] - wall - 0.5])
            translate([bx + win_x[0], y, haube_bot - 0.5])
                cube([win_x[1] - win_x[0], wall + 1, 14 - haube_bot + 0.5]);

        // USB-C links: geschlossenes Fenster abs 13..28.5 (24.08.: unten
        // 10 mm zu — stabiler; Stuelpen kollisionsfrei, Wand liegt 0.75
        // ausserhalb der Steckerspitze)
        translate([-0.5, by + usb_y[0], 13])
            cube([wall + 1, usb_y[1] - usb_y[0], 15.5]);


        // Rechte Stirnwand: NUR Qwiic, geschlossenes Fenster abs 13..27
        translate([outerd[0] - wall - 0.5, by + qwiic_y[0], 13])
            cube([wall + 1, qwiic_y[1] - qwiic_y[0], 14]);

        // Lueftung oben, beide Laengswaende
        for (y = [-0.5, outerd[1] - wall - 0.5], x = [8 : 8 : 68])
            translate([wall + x, y, z_in_top - 6])
                cube([3.5, wall + 1, 4]);

        // LED-Gitter: x 29.5..58.5 (+5 rechts); unterste STREBE schliesst
        // buendig mit dem GPIO-Schlitz Reihe B ab (Steg 6.5..7.9, dann Schlitze)
        for (i = [0 : 3])
            translate([bx + 29.5, by + 7.9 + i * 4.0, z_in_top - 0.5])
                cube([29, 2.6, wall + 1]);

        // JSPI, 90° gedreht (24.08.): hochkant 7.8 x 12.2, Oberkante
        // buendig am Saeulenfuss (y 32.8)
        translate([bx + 62.2, by + 20.6, z_in_top - 0.5]) cube([7.8, 12.2, wall + 1]);

        // JCTL (Overlay: x 17.5..27, y 42..49) — verschmilzt mit Reihe A;
        // startet erst bei x 18.2, damit die Saeule (15.24/50.8) Fuss behaelt
        // buendig bis an die Saeule (x ab 12.4 = Tangente, y bis 48.0 = Saeulenfuss)
        translate([bx + 12.4, by + 40, z_in_top - 0.5]) cube([16.3, 8.0, wall + 1]);

        // Power-Knopf (FOTO: Oberseite, neben USB-C, ~x6.5/y48): Fingerloch
        translate([bx + 6.5, by + 48, z_in_top - 0.5])
            cylinder(d = 10, h = wall + 1);

        // Header-Zugriffsschlitze in der Deckplatte, beide Laengskanten —
        // mit Steg an der jeweiligen Saeulen-/Lochposition
        // Reihe y0 mit Steg um die Saeule (66.04/7.62) — wie Reihe A:
        translate([bx + 18.5, by - 0.7, z_in_top - 0.5]) cube([47.3, 7.2, wall + 1]);
        translate([bx + 69.6, by - 0.7, z_in_top - 0.5]) cube([2.4, 7.2, wall + 1]);  // Reststueck bis zur Wand
        // Reihe y-max (Steg um x=15.24):
        translate([bx + 5,    by + 46.8, z_in_top - 0.5]) cube([6,  7.2, wall + 1]);
        translate([bx + 18.04, by + 46.8, z_in_top - 0.5]) cube([53.96, 7.2, wall + 1]);  // buendig ab Saeule, bis zur Wand

        // M3-Durchgang zu den Boden-Piloten (abs z 5.5)
        for (y = [-0.5, outerd[1] - wall - 0.5])
            translate([bx + 62, y, tray_t + 2.5])
                rotate([-90, 0, 0]) cylinder(d = 3.2, h = wall + 1);

        // Schnapp-Fenster in den Stirnwaenden (Nocke 5.2..6.8 →
        // Fenster 5.0..7.8: 0.2 Verriegelungsspiel, oben Einfuehr-Luft)
        for (yr = [5, 50])
            translate([-0.5, by + yr - 2.6, 5.0]) cube([wall + 1, 5.2, 2.8]);
        for (yr = [4, 50])
            translate([outerd[0] - wall - 0.5, by + yr - 2.6, 5.0]) cube([wall + 1, 5.2, 2.8]);

        // Zip-Tie-Slots fuer wall_tab
        for (y = [-0.5, outerd[1] - wall - 0.5], x = [28, 50])
            translate([wall + x, y, 16]) cube([4, wall + 1, 6]);
    }
    // Niederhalter mit Zentrier-Zapfen in den Ø3.2-Loechern.
    // Saeule am Power-Schalter (15.24/50.8) 24.08. um 2 mm Richtung
    // GPIO-Reihe verschoben (52.8) → steht neben dem Loch, daher OHNE
    // Zapfen (wuerde sonst auf der Platine aufstehen); die Zentrierung
    // uebernehmen die drei verbleibenden Zapfen.
    np = [[13.97, 2.54, 1], [15.24, 52.8, 0], [66.04, 7.62, 1], [66.04, 35.56, 1]];
    for (h = np)
        translate([bx + h[0], by + h[1], z_unoq]) {
            cylinder(d = 5.6, h = z_in_top - z_unoq);
            if (h[2] == 1) translate([0, 0, -2.2]) cylinder(d = 2.6, h = 2.3);
        }
}

// ── Wall-Tab (unveraendert) ───────────────────────────────────
module wall_tab() {
    difference() {
        cube([34, 30, 4]);
        for (x = [5, 27])
            translate([x, -0.5, 1.2]) cube([4, 31, 6]);
    }
    translate([17, 30.1, 0]) gp_male();
}

// ── Auswahl ───────────────────────────────────────────────────
if (part == "boden") boden();
// Haube gedreht exportiert: Deckplatte auf dem Bett, offene
// Ausschnitte zeigen nach oben → support-frei
if (part == "haube") translate([0, outerd[1], H]) rotate([180, 0, 0]) haube();
if (part == "wall_tab") wall_tab();
if (part == "preview") {
    boden();
    translate([0, outerd[1] + 15, 0]) translate([0, outerd[1], H]) rotate([180, 0, 0]) haube();
    translate([outerd[0] + 15, 0, 0]) wall_tab();
}
