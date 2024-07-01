import sys
#importamos el modulo cplex
import cplex
from recordclass import recordclass
import numpy as np
import itertools

Orden = recordclass('Orden', 'id beneficio cant_trab')

class InstanciaAsignacionCuadrillas:
    def __init__(
        self, 
        activar_restriccion_opcional_1 = False, 
        activar_restriccion_opcional_2 = False,
        tolerancia = 1e-6,
        seleccion_nodo = None, 
        seleccion_variable = None,
        heuristica_primal = None,
        preproceso = None,
        penalizacion_conflicto = int(0)
    ):
        self.activar_restriccion_opcional_1 = activar_restriccion_opcional_1
        self.activar_restriccion_opcional_2 = activar_restriccion_opcional_2
        self.penalizacion_conflicto = int(penalizacion_conflicto)
        self.seleccion_nodo = seleccion_nodo
        self.seleccion_variable = seleccion_variable
        self.tolerancia = tolerancia
        self.heuristica_primal = heuristica_primal
        self.preproceso = preproceso

        self.cantidad_trabajadores = 0
        self.cantidad_ordenes = 0
        self.ordenes = []
        self.conflictos_trabajadores = []
        self.ordenes_correlativas = []
        self.ordenes_conflictivas = []
        self.ordenes_repetitivas = []
        self._indices_A_ijd = [] # 6 * T * O variables. Representa si el trabajador i trabaja en la orden j en el dia d
        self._indices_B_jdk = [] # 30 * O variables. Representa si la orden j en el turno k del dia d
        self._indices_TR_id = [] # 6 * T variables. Representa si el trabajador i trabaja en el dia d
        self._indices_delta_j = [] # O variables. Representa si la orden j se asigna a algún turno
        self._indices_x_ir = [] # 4 * T variables. Representa la cantidad de turnos que trabaja el trabajador i en el tramo r
        self._indices_w_ir = [] # 3 * T variables. Representa si el tramo r del trabajador i es activado
        self._indices_Tc_pj = []
        self._total_variables = 0
        self.tiempo_de_computo = 0
        self.funcion_objetivo = 0
        
    def leer_datos(self,nombre_archivo):

        # Se abre el archivo
        f = open(nombre_archivo)

        # Lectura cantidad de trabajadores
        self.cantidad_trabajadores = int(f.readline())
        
        # Lectura cantidad de ordenes
        self.cantidad_ordenes = int(f.readline())

        # Lectura de las ordenes
        self.ordenes = []
        for i in range(self.cantidad_ordenes):
            linea = f.readline().rstrip().split(' ')
            self.ordenes.append(Orden(linea[0],linea[1],linea[2]))
        
        # Lectura cantidad de conflictos entre los trabajadores
        cantidad_conflictos_trabajadores = int(f.readline())
        
        # Lectura conflictos entre los trabajadores
        self.conflictos_trabajadores = []
        for i in range(cantidad_conflictos_trabajadores):
            linea = f.readline().split(' ')
            self.conflictos_trabajadores.append(list(map(int,linea)))
            
        # Lectura cantidad de ordenes correlativas
        cantidad_ordenes_correlativas = int(f.readline())
        
        # Lectura ordenes correlativas
        self.ordenes_correlativas = []
        for i in range(cantidad_ordenes_correlativas):
            linea = f.readline().split(' ')
            self.ordenes_correlativas.append(list(map(int,linea)))
            
        # Lectura cantidad de ordenes conflictivas
        cantidad_ordenes_conflictivas = int(f.readline())
        
        # Lectura ordenes conflictivas
        self.ordenes_conflictivas = []
        for i in range(cantidad_ordenes_conflictivas):
            linea = f.readline().split(' ')
            self.ordenes_conflictivas.append(list(map(int,linea)))
        
        # Lectura cantidad de ordenes repetitivas
        cantidad_ordenes_repetitivas = int(f.readline())
        
        # Lectura ordenes repetitivas
        self.ordenes_repetitivas = []
        for i in range(cantidad_ordenes_repetitivas):
            linea = f.readline().split(' ')
            self.ordenes_repetitivas.append(list(map(int,linea)))
        # Se cierra el archivo de entrada
        f.close()

        # Precálculo de índices
        indice_comienzo = 0
        self._indices_A_ijd = np.arange(
             6 * self.cantidad_trabajadores * self.cantidad_ordenes
        ).reshape(self.cantidad_trabajadores, self.cantidad_ordenes, 6).tolist()
        indice_comienzo += 6 * self.cantidad_trabajadores * self.cantidad_ordenes
        self._indices_B_jdk = (
            np.arange(self.cantidad_ordenes * 6 * 5)
                .reshape(self.cantidad_ordenes, 6, 5) + indice_comienzo
            ).tolist()
        indice_comienzo += self.cantidad_ordenes * 6 * 5
        self._indices_TR_id = (
            np.arange(self.cantidad_trabajadores * 6)
            .reshape(self.cantidad_trabajadores, 6) + indice_comienzo
        ).tolist()
        indice_comienzo += self.cantidad_trabajadores * 6
        self._indices_delta_j = (np.arange(self.cantidad_ordenes) + indice_comienzo).tolist()
        indice_comienzo += self.cantidad_ordenes
        self._indices_x_ir = (
            np.arange(self.cantidad_trabajadores * 4)
            .reshape(self.cantidad_trabajadores, 4) + indice_comienzo
        ).tolist()
        indice_comienzo += self.cantidad_trabajadores * 4
        self._indices_w_ir = (
            np.arange(self.cantidad_trabajadores * 3)
            .reshape(self.cantidad_trabajadores, 3) + indice_comienzo
        ).tolist()
        indice_comienzo += len(self.conflictos_trabajadores)
#variables para analisis
        

        indice_comienzo = indice_comienzo + self.cantidad_trabajadores * 3

        self._indices_Tc_pj = (
            np.arange(len(self.conflictos_trabajadores)*self.cantidad_ordenes)
            .reshape(len(self.conflictos_trabajadores), self.cantidad_ordenes) + indice_comienzo
        ).tolist()

        self._total_variables = indice_comienzo + len(self.conflictos_trabajadores) * self.cantidad_ordenes

def cargar_instancia():
    # El 1er parametro es el nombre del archivo de entrada 	
    nombre_archivo = sys.argv[1].strip()
    # Crea la instancia vacia
    instancia = InstanciaAsignacionCuadrillas()
    # Llena la instancia con los datos del archivo de entrada 
    instancia.leer_datos(nombre_archivo)
    return instancia

def cargar_instancia_con_configuracion(
    path, 
    activar_restriccion_opcional_1 = False, 
    activar_restriccion_opcional_2 = False,
    tolerancia = 1e-6,
    seleccion_nodo = None,
    seleccion_variable = None,
    heuristica_primal = None,
    preproceso = None,
    penalizacion_conflicto = 0
):
    instancia = InstanciaAsignacionCuadrillas(
        activar_restriccion_opcional_1 = activar_restriccion_opcional_1,
        activar_restriccion_opcional_2 = activar_restriccion_opcional_2,
        tolerancia = tolerancia,
        seleccion_nodo = seleccion_nodo,
        seleccion_variable = seleccion_variable,
        heuristica_primal = heuristica_primal,
        preproceso = preproceso,
        penalizacion_conflicto= penalizacion_conflicto
    )
    instancia.leer_datos(path)
    return instancia

def agregar_variables(prob, instancia):
    coeficientes_funcion_objetivo = [0]*instancia._total_variables
    # Beneficios de las ordenes
    for j, d, k in itertools.product(range(instancia.cantidad_ordenes), range(6), range(5)):
        coeficientes_funcion_objetivo[instancia._indices_B_jdk[j][d][k]] = int(instancia.ordenes[j].beneficio)

    # Costos de las ordenes
    for i in range(instancia.cantidad_trabajadores):
        coeficientes_funcion_objetivo[instancia._indices_x_ir[i][0]] = -1000
        coeficientes_funcion_objetivo[instancia._indices_x_ir[i][1]] = -1200
        coeficientes_funcion_objetivo[instancia._indices_x_ir[i][2]] = -1400
        coeficientes_funcion_objetivo[instancia._indices_x_ir[i][3]] = -1500

    
    for p, j in itertools.product(range(len(instancia.conflictos_trabajadores)), range(instancia.cantidad_ordenes)):
        coeficientes_funcion_objetivo[instancia._indices_Tc_pj[p][j]] = -instancia.penalizacion_conflicto

    # Ponemos nombre a las variables
    nombres = [""] * instancia._total_variables

    for i, j, d in itertools.product(range(instancia.cantidad_trabajadores), range(instancia.cantidad_ordenes), range(6)):
        nombres[instancia._indices_A_ijd[i][j][d]] = "A_{}_{}_{}".format(i, j, d)
    
    for j, d, k in itertools.product(range(instancia.cantidad_ordenes), range(6), range(5)):
        nombres[instancia._indices_B_jdk[j][d][k]] = "B_{}_{}_{}".format(j, d, k)
    
    for i, d in itertools.product(range(instancia.cantidad_trabajadores), range(6)):
        nombres[instancia._indices_TR_id[i][d]] = "TR_{}_{}".format(i, d)
    
    for j in range(instancia.cantidad_ordenes):
        nombres[instancia._indices_delta_j[j]] = "delta_{}".format(j)

    for i, r in itertools.product(range(instancia.cantidad_trabajadores), range(4)):
        nombres[instancia._indices_x_ir[i][r]] = "x_{}_{}".format(i, r)
    
    for i, r in itertools.product(range(instancia.cantidad_trabajadores), range(3)):
        nombres[instancia._indices_w_ir[i][r]] = "w_{}_{}".format(i, r)

    for p,j in itertools.product(
            range(len(instancia.conflictos_trabajadores)),
            range(instancia.cantidad_ordenes)):
        nombres[instancia._indices_Tc_pj[p][j]] = "T_{}_{}".format(p, j)
    
    lb = [0] * instancia._total_variables
    ub = [1] * instancia._total_variables

    for i in range(instancia.cantidad_trabajadores):
        for r in range(4):
            ub[instancia._indices_x_ir[i][r]] = instancia.cantidad_ordenes
    
    types = ["B"] * instancia._total_variables

    for i, r in itertools.product(range(instancia.cantidad_trabajadores), range(4)):
        types[instancia._indices_x_ir[i][r]] = "I"
    

    # Agregar las variables
    prob.variables.add(obj = coeficientes_funcion_objetivo, lb = lb, ub = ub, types=types, names=nombres)

def agregar_restricciones(prob, instancia):
    filas = []
    senses = []
    rhs = []
    names = []

    for j in range(instancia.cantidad_ordenes):
        indices = np.reshape(instancia._indices_B_jdk[j], newshape=-1).tolist()
        indices.append(instancia._indices_delta_j[j])
        valores = [1] * (6 * 5) + [-1]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('E')
        rhs.append(0)
        names.append(f"Orden {j} a lo sumo un turno (i)")

    for j in range(instancia.cantidad_ordenes):
        indices = np.reshape(np.array(instancia._indices_A_ijd)[:, j, :], newshape=-1).tolist()
        indices.append(instancia._indices_delta_j[j])
        valores = [1] * (6 * instancia.cantidad_trabajadores) + [-int(instancia.ordenes[j].cant_trab)]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('E')
        rhs.append(0)
        names.append(f"Orden {j} respeta cantidad de trabajadores, si se realiza")

    for i, j in itertools.product(range(instancia.cantidad_trabajadores), range(instancia.cantidad_ordenes)):
        indices = np.reshape(np.array(instancia._indices_A_ijd)[i, j, :], newshape=-1).tolist()
        valores = [1] * 6
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(1)
        names.append(f"Trabajador {i} trabaja en la orden {j} a lo sumo un dia")

    for i, (j1, j2), d, k in itertools.product(
        range(instancia.cantidad_trabajadores),
        itertools.combinations(range(instancia.cantidad_ordenes), 2),
        range(6),
        range(5)
    ):
        indices = [
            instancia._indices_A_ijd[i][j1][d],
            instancia._indices_B_jdk[j1][d][k],
            instancia._indices_A_ijd[i][j2][d],
            instancia._indices_B_jdk[j2][d][k]
        ]
        valores = [1, 1, 1, 1]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(3)
        names.append(f"Trabajador {i} no puede trabajar en las ordenes {j1} y {j2} en el turno {k} del dia {d}")
    
    #AGREGAR AL MODELO
    for i, j, d in itertools.product(
        range(instancia.cantidad_trabajadores),
        range(instancia.cantidad_ordenes),
        range(6),
    ):
        indices = [
            instancia._indices_A_ijd[i][j][d],
            *instancia._indices_B_jdk[j][d],
        ]
        valores = [1] + [-1] * 5
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(0)
        names.append(f"Si el trabajador {i} trabaja en la orden {j} en el dia {d}, entonces la orden es asignada al día {d}")

    for i, d in itertools.product(range(instancia.cantidad_trabajadores), range(6)):
        indices = np.reshape(np.array(instancia._indices_A_ijd)[i, :, d], newshape=-1).tolist()
        indices.append(instancia._indices_TR_id[i][d])
        valores = [1] * instancia.cantidad_ordenes + [-4]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(0)
        names.append(f"Trabajador {i} no trabaja los 5 turnos")

    for i, d in itertools.product(range(instancia.cantidad_trabajadores), range(6)):
        indices = np.reshape(np.array(instancia._indices_A_ijd)[i, :, d], newshape=-1).tolist()
        indices.append(instancia._indices_TR_id[i][d])
        valores = [1] * instancia.cantidad_ordenes + [-1]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('G')
        rhs.append(0)
        names.append(f"Activacion Tr_{i}_{d}")

    for i in range(instancia.cantidad_trabajadores):
        indices = instancia._indices_TR_id[i]
        valores = [1] * 6
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(5)
        names.append(f"Trabajador {i} no trabaja todos los dias")

    # Ordenes conflictivas
    for (j1, j2), i, d, k in itertools.product(
        instancia.ordenes_conflictivas, 
        range(instancia.cantidad_trabajadores), 
        range(6), 
        range(4) # No considerar ultimo turno
    ):
        indices = [
            instancia._indices_A_ijd[i][j1][d], 
            instancia._indices_B_jdk[j1][d][k],
            instancia._indices_A_ijd[i][j2][d],
            instancia._indices_B_jdk[j2][d][k+1]
        ]
        valores = [1, 1, 1, 1]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(3)
        names.append(f"Ordenes {j1} y {j2} no pueden ser asignadas al mismo trabajador consecutivamente")

        indices = [
            instancia._indices_A_ijd[i][j2][d], 
            instancia._indices_B_jdk[j2][d][k],
            instancia._indices_A_ijd[i][j1][d],
            instancia._indices_B_jdk[j1][d][k+1]
        ]
        valores = [1, 1, 1, 1]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(3)
        names.append(f"Ordenes {j2} y {j1} no pueden ser asignadas al mismo trabajador consecutivamente")

    for (j1, j2), d, k in itertools.product(
        instancia.ordenes_correlativas, 
        range(6), 
        range(4)
    ):
        indices = [
            instancia._indices_B_jdk[j1][d][k],
            instancia._indices_B_jdk[j2][d][k+1]
        ]
        valores = [1, -1]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(0)
        names.append(f"Si {j1} en {k}, entonces {j2} en {k+1}")
    
    #n

    for (j1, j2), d in itertools.product(
        instancia.ordenes_correlativas, 
        range(6)
        ):
        indices = [
            instancia._indices_B_jdk[j1][d][4],
            instancia._indices_B_jdk[j2][d][4]
        ]
        valores = [1, 1]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(0)
        names.append(f" {j1} y {j2} no pueden estar en el ultimo turno")
    

    for (i1, i2) in itertools.combinations(range(instancia.cantidad_trabajadores), 2):
        indices_1 = np.reshape(np.array(instancia._indices_A_ijd)[i1, :, :], newshape=-1)
        indices_2 = np.reshape(np.array(instancia._indices_A_ijd)[i2, :, :], newshape=-1)
        indices = np.concatenate([indices_1, indices_2]).tolist()
        valores = [1] * instancia.cantidad_ordenes * 6 + [-1] * instancia.cantidad_ordenes * 6

        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(8)
        names.append(f"No puede haber una diferencia mayor a 8 turnos entre los trabajadores {i1} y {i2}")

        fila = [indices,valores]
        filas.append(fila)
        senses.append('G')
        rhs.append(-8)
        names.append(f"No puede haber una diferencia mayor a 8 turnos entre los trabajadores {i1} y {i2}")

    for i in range(instancia.cantidad_trabajadores):
        indices = np.concatenate([
            np.reshape(np.array(instancia._indices_A_ijd)[i, :, :], newshape=-1),
            np.array(instancia._indices_x_ir)[i, :]
        ]).tolist()
        valores = [1] * instancia.cantidad_ordenes * 6 + [-1] * 4
        fila = [indices,valores]
        filas.append(fila)
        senses.append('E')
        rhs.append(0)
        names.append(f"Restriccion de la cantidad de turnos para el trabajador {i}")

        indices = [instancia._indices_x_ir[i][0], instancia._indices_w_ir[i][0]]
        valores = [1, -5]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('G')
        rhs.append(0)
        names.append(f"Restriccion primer tramo funcion de costo trabajador (i) {i}")

        indices = [instancia._indices_x_ir[i][0]]
        valores = [1]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(5)
        names.append(f"Restriccion primer tramo funcion de costo trabajador (ii) {i}")

        indices = [instancia._indices_x_ir[i][1], instancia._indices_w_ir[i][1]]
        valores = [1, -5]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('G')
        rhs.append(0)
        names.append(f"Restriccion segundo tramo funcion de costo trabajador (i) {i}")

        indices = [instancia._indices_x_ir[i][1], instancia._indices_w_ir[i][0]]
        valores = [1, -5]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(0)
        names.append(f"Restriccion segundo tramo funcion de costo trabajador (ii) {i}")

        indices = [instancia._indices_x_ir[i][2], instancia._indices_w_ir[i][2]]
        valores = [1, -5]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('G')
        rhs.append(0)
        names.append(f"Restriccion tercer tramo funcion de costo trabajador (i) {i}")

        indices = [instancia._indices_x_ir[i][2], instancia._indices_w_ir[i][1]]
        valores = [1, -5]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(0)
        names.append(f"Restriccion tercer tramo funcion de costo trabajador (ii) {i}")
        
        indices = [instancia._indices_x_ir[i][3], instancia._indices_w_ir[i][2]]
        valores = [1, -5]
        fila = [indices,valores]
        filas.append(fila)
        senses.append('L')
        rhs.append(0)
        names.append(f"Restriccion cuarto tramo funcion de costo trabajador {i}")

    if instancia.penalizacion_conflicto > 0:
        for p, j, d in itertools.product(
            range(len(instancia.conflictos_trabajadores)),
            range(instancia.cantidad_ordenes),
            range(6)
        ):
            i1, i2 = instancia.conflictos_trabajadores[p]
            indices = [
                instancia._indices_Tc_pj[p][j],
                instancia._indices_A_ijd[i1][j][d],
                instancia._indices_A_ijd[i2][j][d]
            ]
            valores = [-1, 1, 1]
            fila = [indices,valores]
            filas.append(fila)
            senses.append('L')
            rhs.append(1)
            names.append(f"Activación Tc_{p}_{j} ({i1} - {i2})")

    # Restricción adicional 1 
    if instancia.activar_restriccion_opcional_1:
        for (i1,i2), j, d in itertools.product(
            instancia.conflictos_trabajadores,
            range(instancia.cantidad_ordenes),
            range(6)
        ):
            indices = [
                instancia._indices_A_ijd[i1][j][d],
                instancia._indices_A_ijd[i2][j][d],
            ]
            valores = [1, 1]
            fila = [indices,valores]
            filas.append(fila)
            senses.append('L')
            rhs.append(1)
            names.append(f"Trabajador {i1} y trabajador {i2} no trabajan en una misma orden si estan conflictuados")

    # Restricción adicional 2
    
    if instancia.activar_restriccion_opcional_2:
        for i, (j1,j2) in itertools.product(
            range(instancia.cantidad_trabajadores),
            instancia.ordenes_repetitivas
        ):
            indices = [
                *instancia._indices_A_ijd[i][j1],
                *instancia._indices_A_ijd[i][j2],
            ]
            valores = [1] * 6 + [1] * 6
            fila = [indices,valores]
            filas.append(fila)
            senses.append('L')
            rhs.append(1)
            names.append(f"Trabajador {i} no trabaja en las ordenes {j1} y {j2} si son repetitivas")


    prob.linear_constraints.add(lin_expr=filas, senses=senses, rhs=rhs, names=names)


def armar_lp(prob, instancia, shouldWrite = True):

    # Agregar las variables
    agregar_variables(prob, instancia)
   
    # Agregar las restricciones 
    agregar_restricciones(prob, instancia)

    prob.objective.set_sense(prob.objective.sense.maximize)

    # Escribir el lp a archivo
    if shouldWrite:
        prob.write('asignacionCuadrillas.lp')

def resolver_lp(prob, instancia):
    # Definir los parametros del solver
    prob.parameters.mip.tolerances.mipgap.set(instancia.tolerancia)

    if instancia.seleccion_nodo is not None:
        prob.parameters.mip.strategy.search.set(instancia.seleccion_nodo)
    
    if instancia.seleccion_variable is not None:
        prob.parameters.mip.strategy.variableselect.set(instancia.seleccion_variable)
    
    if instancia.heuristica_primal is not None:
        prob.parameters.mip.strategy.heuristiceffort.set(instancia.heuristica_primal)
    
    if instancia.preproceso is not None:
        prob.parameters.preprocessing.presolve.set(instancia.preproceso)
    
    start_time = prob.get_time()

    # Resolver el lp
    prob.solve()

    end_time = prob.get_time()
    instancia.tiempo_de_computo = end_time - start_time
    instancia.funcion_objetivo = prob.solution.get_objective_value()
    

# Obtener informacion de la solucion a traves de 'solution'
def mostrar_solucion(prob, instancia):
    # Tomar el estado de la resolucion
    status = prob.solution.get_status_string(status_code = prob.solution.get_status())
    
    # Tomar el valor del funcional
    valor_obj = prob.solution.get_objective_value()
    
    print('Funcion objetivo: ',valor_obj,'(' + str(status) + ')')
    
    # Tomar los valores de las variables
    x  = prob.solution.get_values()

    # Mostrar las variables con valor positivo (mayor que una tolerancia)

    # Mostramos las asignaciones de la siguiente forma:
    # Trabajador i trabaja en la orden j en el dia d
    for i, j, d in itertools.product(range(instancia.cantidad_trabajadores), range(instancia.cantidad_ordenes), range(6)):
        if x[instancia._indices_A_ijd[i][j][d]] > instancia.tolerancia:
            print(f"Trabajador {i} trabaja en la orden {j} en el dia {d}")

    # La orden j se asigna al turno k del dia d
    for j, d, k in itertools.product(range(instancia.cantidad_ordenes), range(6), range(5)):
        if x[instancia._indices_B_jdk[j][d][k]] > instancia.tolerancia:
            print(f"La orden {j} se asigna al turno {k} del dia {d}")

    # La orden j es realizada por los trabajadores i1, ..., imax / La orden j no se realiza
    for j in range(instancia.cantidad_ordenes):
        trabajadores = []
        for i, d in itertools.product(range(instancia.cantidad_trabajadores), range(6)):
            if x[instancia._indices_A_ijd[i][j][d]] > instancia.tolerancia:
                trabajadores.append(i)
        if x[instancia._indices_delta_j[j]] > instancia.tolerancia:
            print(f"La orden {j} es realizada por los trabajadores {trabajadores}")
        else:
            print(f"La orden {j} no se realiza. Por lo tanto, los trabajadores son {trabajadores}")

def main():
    
    # Lectura de datos desde el archivo de entrada
    instancia = cargar_instancia()
    
    # Definicion del problema de Cplex
    prob = cplex.Cplex()

    # Setear el sentido del problema
    prob.objective.set_sense(prob.objective.sense.maximize)
    
    # Definicion del modelo
    armar_lp(prob,instancia)

    # Resolucion del modelo
    resolver_lp(prob)

    # Obtencion de la solucion
    mostrar_solucion(prob,instancia)

if __name__ == '__main__':
    main()
