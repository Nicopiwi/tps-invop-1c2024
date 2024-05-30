import sys
#importamos el modulo cplex
import cplex
from recordclass import recordclass
import numpy as np
import itertools

TOLERANCE =10e-6 
Orden = recordclass('Orden', 'id beneficio cant_trab')

class InstanciaAsignacionCuadrillas:
    def __init__(self):
        self.cantidad_trabajadores = 0
        self.cantidad_ordenes = 0
        self.ordenes = []
        self.conflictos_trabajadores = []
        self.ordenes_correlativas = []
        self.ordenes_conflictivas = []
        self.ordenes_repetitivas = []
        self._indices_A_ijd = [] # 6 * T * O variables
        self._indices_B_idk = [] # 30 * O variables
        self._indices_TR_id = [] # 6 * T variables
        self._indices_delta_j = [] # O variables
        self._indices_x_ir = [] # 4 * T variables
        self._indices_w_ir = [] # 3 * T variables
        self._total_variables = 0
        
    def leer_datos(self,nombre_archivo):

        # Se abre el archivo
        f = open(nombre_archivo)

        # Lectura cantidad de trabajadores
        self.cantidad_trabajadores = int(f.readline())
        
        # Lectura cantidad de ordenes
        self.cantidad_ordenes = int(f.readline())

        # Precalculo de indices

        indice_comienzo = 0
        self._indices_A_ijd = np.arange(
             6 * self.cantidad_trabajadores * self.cantidad_ordenes
        ).reshape(self.cantidad_trabajadores, self.cantidad_ordenes, 6).tolist()
        indice_comienzo += 6 * self.cantidad_trabajadores * self.cantidad_ordenes
        self._indices_B_idk = (np.arange(self.cantidad_ordenes * 6 * 5).reshape(self.cantidad_ordenes, 6, 5) + indice_comienzo).tolist()
        indice_comienzo += self.cantidad_ordenes * 6 * 5
        self._indices_TR_id = (np.arange(self.cantidad_trabajadores * 6).reshape(self.cantidad_trabajadores, 6) + indice_comienzo).tolist()
        indice_comienzo += self.cantidad_trabajadores * 6
        self._indices_delta_j = (np.arange(self.cantidad_ordenes) + indice_comienzo).tolist()
        indice_comienzo += self.cantidad_ordenes
        self._indices_x_ir = (np.arange(self.cantidad_trabajadores * 4).reshape(self.cantidad_trabajadores, 4) + indice_comienzo).tolist()
        indice_comienzo += self.cantidad_trabajadores * 4
        self._indices_w_ir = (np.arange(self.cantidad_trabajadores * 3).reshape(self.cantidad_trabajadores, 3) + indice_comienzo).tolist()
        self._total_variables = 6 * self.cantidad_trabajadores * self.cantidad_ordenes + 5 * self.cantidad_ordenes + 6 * self.cantidad_trabajadores + self.cantidad_ordenes + 4 * self.cantidad_trabajadores + 3 * self.cantidad_trabajadores

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


def cargar_instancia():
    # El 1er parametro es el nombre del archivo de entrada 	
    nombre_archivo = sys.argv[1].strip()
    # Crea la instancia vacia
    instancia = InstanciaAsignacionCuadrillas()
    # Llena la instancia con los datos del archivo de entrada 
    instancia.leer_datos(nombre_archivo)
    return instancia

def agregar_variables(prob, instancia):
    # Llenar coef\_funcion\_objetivo
    coeficientes_funcion_objetivo = [0]*instancia._total_variables

    # Llenar coeficientes_funcion_objetivo con los beneficios de las ordenes
    for j, d, k in itertools.product(range(instancia.cantidad_ordenes), range(6), range(5)):
        coeficientes_funcion_objetivo[instancia._indices_B_jdk[j][d][k]] = instancia.ordenes[j].beneficio

    # Llenar coeficientes_funcion_objetivo con los costos de las ordenes
    for i in range(instancia.cantidad_trabajadores):
        coeficientes_funcion_objetivo[instancia._indices_x_it[i][0]] = -1000
        coeficientes_funcion_objetivo[instancia._indices_x_it[i][0]] = -1200
        coeficientes_funcion_objetivo[instancia._indices_x_it[i][0]] = -1400
        coeficientes_funcion_objetivo[instancia._indices_x_it[i][0]] = -1500

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
    
    lb = [0] * instancia._total_variables
    ub = [1] * instancia._total_variables

    for i in range(instancia.cantidad_trabajadores):
        for r in range(4):
            ub[instancia._indices_x_ir[i][r]] = instancia.cantidad_ordenes
    
    types = ["B"] * instancia._total_variables

    for i, r in itertools.product(range(instancia.cantidad_trabajadores), range(4)):
        types[instancia._indices_x_ir[0][0]] = "I"
    
    # Agregar las variables
    prob.variables.add(obj = coeficientes_funcion_objetivo, lb = lb, ub = ub, types=types, names=nombres)

def agregar_restricciones(prob, instancia):
    # Agregar las restricciones ax <= (>= ==) b:
	# funcion 'add' de 'linear_constraints' con parametros:
	# lin_expr: lista de listas de [ind,val] de a
    # sense: lista de 'L', 'G' o 'E'
    # rhs: lista de los b
    # names: nombre (como van a aparecer en el archivo .lp)
	
    # Notar que cplex espera "una matriz de restricciones", es decir, una
    # lista de restricciones del tipo ax <= b, [ax <= b]. Por lo tanto, aun cuando
    # agreguemos una unica restriccion, tenemos que hacerlo como una lista de un unico
    # elemento.

    # Restriccion generica
    indices = ...
    valores = ...
    fila = [indices,valores]
    prob.linear_constraints.add(lin_expr=[fila], senses=[...], rhs=[...], names=[...])

def armar_lp(prob, instancia):

    # Agregar las variables
    agregar_variables(prob, instancia)
   
    # Agregar las restricciones 
    agregar_restricciones(prob, instancia)

    # Setear el sentido del problema
    prob.objective.set_sense(prob.objective.sense.....)

    # Escribir el lp a archivo
    prob.write('asignacionCuadrillas.lp')

def resolver_lp(prob):
    
    # Definir los parametros del solver
    prob.parameters....
       
    # Resolver el lp
    prob.solve()

#def mostrar_solucion(prob,instancia):
    # Obtener informacion de la solucion a traves de 'solution'
    
    # Tomar el estado de la resolucion
    status = prob.solution.get_status_string(status_code = prob.solution.get_status())
    
    # Tomar el valor del funcional
    valor_obj = prob.solution.get_objective_value()
    
    print('Funcion objetivo: ',valor_obj,'(' + str(status) + ')')
    
    # Tomar los valores de las variables
    x  = prob.solution.get_values()
    # Mostrar las variables con valor positivo (mayor que una tolerancia)
    .....

def main():
    
    # Lectura de datos desde el archivo de entrada
    instancia = cargar_instancia()
    
    # Definicion del problema de Cplex
    prob = cplex.Cplex()
    
    # Definicion del modelo
    armar_lp(prob,instancia)

    # Resolucion del modelo
    resolver_lp(prob)

    # Obtencion de la solucion
    mostrar_solucion(prob,instancia)

if __name__ == '__main__':
    main()
