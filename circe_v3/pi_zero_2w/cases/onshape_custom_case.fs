FeatureScript 3070;
import(path : "onshape/std/geometry.fs", version : "3070.0");

// Raspberry Pi Zero 2 W enclosure (base + lid), all dimensions in mm.
// Board data from raspberry-pi-zero-2-w-mechanical-drawing.pdf:
//   outline 65 x 30; mounting holes 3.5 in from every edge (58 x 23 pattern);
//   mini-HDMI CL x=12.4; micro-USB OTG CL x=41.4; micro-USB PWR CL x=54.
// Verified: test_feature evaluated with zero notices; geometry built in Onshape.

annotation { "Feature Type Name" : "PiZero2WCase" }
export const PiZero2WCase = defineFeature(function(context is Context, id is Id, definition is map)
    precondition
    {
        annotation { "Name" : "Wall thickness" }
        isLength(definition.wall, LENGTH_BOUNDS);
        annotation { "Name" : "Board clearance" }
        isLength(definition.clearance, LENGTH_BOUNDS);
        annotation { "Name" : "Standoff height" }
        isLength(definition.standoff, LENGTH_BOUNDS);
        annotation { "Name" : "Headroom above PCB" }
        isLength(definition.headroom, LENGTH_BOUNDS);
        annotation { "Name" : "Lid thickness" }
        isLength(definition.lidThickness, LENGTH_BOUNDS);
    }
    {
        const mm = millimeter;
        const wall = definition.wall / mm;
        const clr = definition.clearance / mm;
        const so = definition.standoff / mm;
        const head = definition.headroom / mm;
        const lidT = definition.lidThickness / mm;
        const PCB_L = 65.0;
        const PCB_W = 30.0;
        const PCB_T = 1.4;
        const floorT = 2.0;

        const ox = wall + clr;
        const oy = wall + clr;
        const innerL = PCB_L + 2 * clr;
        const innerW = PCB_W + 2 * clr;
        const outerL = innerL + 2 * wall;
        const outerW = innerW + 2 * wall;
        const zPcb = floorT + so;
        const zTop = zPcb + PCB_T;
        const baseH = zTop + head;
        const lipH = 2.0;
        const lipGap = 0.3;
        const lipWall = 1.2;

        // ---- base shell -------------------------------------------------
        fCuboid(context, id + "baseSolid", {
                    "corner1" : vector(0, 0, 0) * mm,
                    "corner2" : vector(outerL, outerW, baseH) * mm
                });
        fCuboid(context, id + "cavity", {
                    "corner1" : vector(wall, wall, floorT) * mm,
                    "corner2" : vector(wall + innerL, wall + innerW, baseH + 1) * mm
                });
        opBoolean(context, id + "hollow", {
                    "tools" : qCreatedBy(id + "cavity", EntityType.BODY),
                    "targets" : qCreatedBy(id + "baseSolid", EntityType.BODY),
                    "operationType" : BooleanOperationType.SUBTRACTION
                });

        // ---- standoffs at the 58 x 23 mounting pattern -------------------
        const holeX = [3.5, 61.5];
        const holeY = [3.5, 26.5];
        var posts = [qCreatedBy(id + "baseSolid", EntityType.BODY)];
        var drills = [];
        for (var i = 0; i < 2; i += 1)
        {
            for (var j = 0; j < 2; j += 1)
            {
                const cx = ox + holeX[i];
                const cy = oy + holeY[j];
                const pName = id + ("post" ~ i ~ j);
                const dName = id + ("drill" ~ i ~ j);
                fCylinder(context, pName, {
                            "bottomCenter" : vector(cx, cy, floorT) * mm,
                            "topCenter" : vector(cx, cy, zPcb) * mm,
                            "radius" : 2.75 * mm
                        });
                posts = append(posts, qCreatedBy(pName, EntityType.BODY));
                fCylinder(context, dName, {
                            "bottomCenter" : vector(cx, cy, -1) * mm,
                            "topCenter" : vector(cx, cy, zPcb + 1) * mm,
                            "radius" : 1.05 * mm
                        });
                drills = append(drills, qCreatedBy(dName, EntityType.BODY));
            }
        }
        opBoolean(context, id + "addPosts", {
                    "tools" : qUnion(posts),
                    "operationType" : BooleanOperationType.UNION
                });

        // ---- port cut-outs (open-topped slots, capped by the lid) --------
        // mini-HDMI, CL x = 12.4
        fCuboid(context, id + "cutHdmi", {
                    "corner1" : vector(ox + 5.4, -1, zPcb - 1) * mm,
                    "corner2" : vector(ox + 19.4, wall + clr + 1, baseH + 1) * mm
                });
        // micro-USB OTG (CL 41.4) + micro-USB PWR (CL 54), merged into one slot
        fCuboid(context, id + "cutUsb", {
                    "corner1" : vector(ox + 35.5, -1, zPcb - 1) * mm,
                    "corner2" : vector(ox + 59.5, wall + clr + 1, baseH + 1) * mm
                });
        // microSD, left edge, socket on the PCB underside
        fCuboid(context, id + "cutSd", {
                    "corner1" : vector(-1, oy + 6, floorT) * mm,
                    "corner2" : vector(wall + clr + 1, oy + 24, baseH + 1) * mm
                });
        // CSI-2 camera FFC, right edge
        fCuboid(context, id + "cutCsi", {
                    "corner1" : vector(outerL - wall - clr - 1, oy + 5, zPcb) * mm,
                    "corner2" : vector(outerL + 1, oy + 25, baseH + 1) * mm
                });
        var portTools = append(drills, qCreatedBy(id + "cutHdmi", EntityType.BODY));
        portTools = append(portTools, qCreatedBy(id + "cutUsb", EntityType.BODY));
        portTools = append(portTools, qCreatedBy(id + "cutSd", EntityType.BODY));
        portTools = append(portTools, qCreatedBy(id + "cutCsi", EntityType.BODY));
        opBoolean(context, id + "cutPorts", {
                    "tools" : qUnion(portTools),
                    "targets" : qCreatedBy(id + "baseSolid", EntityType.BODY),
                    "operationType" : BooleanOperationType.SUBTRACTION
                });

        // ---- lid ---------------------------------------------------------
        fCuboid(context, id + "lidPlate", {
                    "corner1" : vector(0, 0, baseH) * mm,
                    "corner2" : vector(outerL, outerW, baseH + lidT) * mm
                });
        fCuboid(context, id + "lipOuter", {
                    "corner1" : vector(wall + lipGap, wall + lipGap, baseH - lipH) * mm,
                    "corner2" : vector(wall + innerL - lipGap, wall + innerW - lipGap, baseH) * mm
                });
        fCuboid(context, id + "lipInner", {
                    "corner1" : vector(wall + lipGap + lipWall, wall + lipGap + lipWall, baseH - lipH - 1) * mm,
                    "corner2" : vector(wall + innerL - lipGap - lipWall, wall + innerW - lipGap - lipWall, baseH + 1) * mm
                });
        opBoolean(context, id + "lipRim", {
                    "tools" : qCreatedBy(id + "lipInner", EntityType.BODY),
                    "targets" : qCreatedBy(id + "lipOuter", EntityType.BODY),
                    "operationType" : BooleanOperationType.SUBTRACTION
                });
        opBoolean(context, id + "lidJoin", {
                    "tools" : qUnion([
                                qCreatedBy(id + "lidPlate", EntityType.BODY),
                                qCreatedBy(id + "lipOuter", EntityType.BODY)
                            ]),
                    "operationType" : BooleanOperationType.UNION
                });
        // 40-pin GPIO header slot
        fCuboid(context, id + "cutGpio", {
                    "corner1" : vector(ox + 5.5, oy + 23.0, baseH - lipH - 1) * mm,
                    "corner2" : vector(ox + 59.5, wall + innerW, baseH + lidT + 1) * mm
                });
        opBoolean(context, id + "cutLid", {
                    "tools" : qCreatedBy(id + "cutGpio", EntityType.BODY),
                    "targets" : qCreatedBy(id + "lidPlate", EntityType.BODY),
                    "operationType" : BooleanOperationType.SUBTRACTION
                });

    }, {
        wall : 2 * millimeter,
        clearance : 0.5 * millimeter,
        standoff : 3.5 * millimeter,
        headroom : 6 * millimeter,
        lidThickness : 2 * millimeter
    });
