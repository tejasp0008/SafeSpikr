`timescale 1ns/1ps

module tb_mac;
  reg clk, rst;
  reg [7:0] a, b;
  wire [15:0] out;

  mac DUT (
    .clk(clk),
    .rst(rst),
    .a(a),
    .b(b),
    .out(out)
  );

  initial begin
    $dumpfile("mac.vcd");
    $dumpvars(0, tb_mac);

    clk = 0;
    rst = 1;
    a = 0;
    b = 0;
    #10 rst = 0;

    // Test cases
    a = 8'd5; b = 8'd3; #10;
    a = 8'd10; b = 8'd2; #10;
    a = 8'd255; b = 8'd1; #10;

    #20 $finish;
  end

  always #5 clk = ~clk;

endmodule

