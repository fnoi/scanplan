import ifcopenshell
import ifcopenshell.geom as geom

settings = geom.settings()
settings.set(settings.USE_WORLD_COORDS, True)

ifc_file = ifcopenshell.open('../data/ifc/TUMCMS_NavVis_Hall.ifc')
for obj in ifc_file.by_type('IfcProduct'):
    print(obj)
    # shape = ifcopenshell.geom.create_shape(settings, obj)
    # geome = shape.geometry
    #
    # print(geome.verts)
    # print(geome.edges)
    # print(geome.faces)
